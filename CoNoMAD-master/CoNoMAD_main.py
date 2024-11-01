import argparse
import os
import torch
from tqdm import tqdm
import yaml
import net
import data
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import random
from sklearn import metrics
import numpy as np
import torch.nn.functional as F


# from loss import loss_function




parser = argparse.ArgumentParser(description='Configuration Import')
parser.add_argument('--config',
                    default="config/drebin.yaml",
                    type=str,
                    metavar='PATH',
                    required=False,
                    help='Configuration path')
parser.add_argument('--gpus',
                    default="0",
                    type=str,
                    required=False,
                    help='gpus')
parser.add_argument('--noise_ratio', default=0.45, type=float, required=False)
parser.add_argument('--random_seed', default=42, type=int, required=False)
parser.add_argument('--order', default=False, type=bool, required=False, help='训练集样本顺序')
parser.add_argument('--order_type', default=0, type=int, required=False, help='样本顺序类型')
parser.add_argument('--alpha', default=0.4, type=float, required=False)


args = parser.parse_args()
random.seed(args.random_seed)

with open(args.config,"r",encoding="utf-8") as f: 
    config = edict(yaml.load(f, Loader=yaml.FullLoader))


# GPU select
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = config.train.num_epochs
accumulation_steps = config.train.accumulation_steps
batch_size = config.train.batch_size
# model_save_path = config.model.save_path

task_name = config.data.dataset_name + "_" + config.data.data_name +"_noise"+str(args.noise_ratio)\
    +"_model"+config.model.model_name + "_bs"+str(batch_size) + "_epochs"+ str(num_epochs)+"_seed"+str(args.random_seed)\
    +str(args.order)+str(args.order_type)+str(args.alpha)

print(task_name)

work_dir = os.path.join(os.path.dirname(__file__), "record", config.data.dataset_name, config.data.file_name, task_name)
model_save_path = os.path.join(work_dir, "checkpoints")
os.makedirs(model_save_path, exist_ok=True)
model = net.get_model(config.model.model_name, num_classes=len(config.data.target_names), input_dim=config.data.data_dim)
model.to(device)

model_best = net.get_model(config.model.model_name, num_classes=len(config.data.target_names), input_dim=config.data.data_dim)
model_best.to(device)


def softmax_temperature(logits, temperature):
    return torch.softmax(logits / temperature, dim=1)

def loss_function(preds, true_labels, pseudo_labels, alpha):

    softmax_loss = F.cross_entropy(preds, true_labels)

    # Calculating KL Scatter
    # pseudo_probs = F.softmax(pseudo_labels, dim=1)
    # pseudo_probs = F.softmax(pseudo_labels, dim=1)
    pseudo_probs = softmax_temperature(pseudo_labels, 8)

    log_preds = F.log_softmax(preds, dim=1)
    kl_div_loss = F.kl_div(log_preds, pseudo_probs, reduction='batchmean')

    return alpha*softmax_loss + (1-alpha)*kl_div_loss


def generate_pseudo_labels(model, data_loader):
    pseudo_labels = []
    # cluster_labels = [[]*10]
    model.eval()
    with torch.no_grad():
        for i, (X, _, cluster_labels) in enumerate(data_loader):
            X = X.to(device)
            outputs, _ = model(X)

            pseudo_labels.append(outputs)
    return pseudo_labels


train_dataset, val_dataset = data.get_dataset(config=config.data, noise_ratio=args.noise_ratio, order=args.order, order_type=args.order_type)

print("len(train_dataset)", len(train_dataset))
train_loader_ori= data.get_dataLoader(config.data, train_dataset)


best_model_path = r"H:\2304noise\toy_deeplearning\ori_deep\record\{}\{}\{}_{}_noise{}_modelDNN_bs10_epochs50_seed42False0\checkpoints\best_checkpoint.pkl"
best_model_path = best_model_path.format("CIC2020_8076", config.data.ori_model_name, "CIC2020_8076", config.data.data_name, str(args.noise_ratio))

print("load model", best_model_path)


state_dict = torch.load(best_model_path)
model_best.load_state_dict(state_dict)
model_best.eval()

pseudo_labels = generate_pseudo_labels(model_best,train_loader_ori)

# train_dataset_pseudo, val_dataset = data.get_dataset(config=config.data, noise_ratio=args.noise_ratio, order=args.order, order_type=args.order_type, pseudo_labels=pseudo_labels)

# train_loader= data.get_dataLoader(config.data, train_dataset_pseudo)

val_loader= data.get_dataLoader(config.data, val_dataset)

loss_func_val = torch.nn.CrossEntropyLoss()
# loss_func = loss.loss_function()

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=config.train.learning_rate, weight_decay=0.0005)

writer = SummaryWriter(os.path.join(work_dir,"Summary"))
global global_step
global_step = 0

def train():
    def train_one_epoch():
        global global_step
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        bar = tqdm(zip(train_loader_ori, pseudo_labels))
        for i,((X_train, y_train_ori, cluster_labels), pseudo_label_batch) in enumerate(bar):
            X_train, y_train, y_train_ori = X_train.to(device), pseudo_label_batch.to(device), y_train_ori.to(device)
            cluster_labels = cluster_labels.to(device)
            assert y_train.size(0) == y_train_ori.size(0)
            outputs, outputs2 = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            loss = loss_function(outputs, y_train_ori, y_train, args.alpha)
            loss2 = loss_func_val(outputs2, cluster_labels)
            t = 0.5
            loss = (1-t)*loss + t*loss2
            # print(i, loss)
            # loss = loss/accumulation_steps
            loss.backward()
            train_loss += loss.item()
            train_correct += torch.sum(pred == y_train_ori.data)
            if((i+1)%accumulation_steps)==0:
                optimizer.step()
                optimizer.zero_grad()
                train_acc = train_correct/((i+1)*batch_size)
                train_loss_ = train_loss/((i+1)*batch_size)
                bar.set_description(f'Batch {i}/{len(train_loader_ori)}')
                bar.set_postfix(loss=train_loss_, correct=train_acc.cpu().numpy())
                writer.add_scalar(tag="loss/train", scalar_value=train_loss_, global_step=global_step)
                writer.add_scalar(tag="acc/train", scalar_value=train_acc, global_step=global_step)
                global_step = global_step + 1

        avg_train_loss = train_loss / len(train_dataset)
        avg_train_acc = 100. * float(train_correct) / float(len(train_dataset))
        return  avg_train_loss, avg_train_acc
    
    
    def validation():
        global global_step
        model.eval()
        val_loss = 0
        val_correct = 0
        # global global_step
        with torch.no_grad():
            for i, (X_val, y_val) in enumerate(val_loader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs, _ = model(X_val)
                _, pred = torch.max(outputs.data, 1)
                val_correct += torch.sum(pred == y_val.data)
                loss = loss_func_val(outputs, y_val)
                val_loss += loss.item()
            val_loss /= len(val_dataset)
            val_accuracy = 100. * float(val_correct) / float(len(val_dataset))
            writer.add_scalar(tag="loss/val", scalar_value=val_loss, global_step=global_step)
            writer.add_scalar(tag="acc/val", scalar_value=val_accuracy, global_step=global_step)
            global_step+=1
            print('Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
                val_loss, val_correct, len(val_dataset), val_accuracy))
            return val_loss, val_accuracy
    best_val_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        avg_train_loss, avg_train_acc = train_one_epoch()
        avg_val_loss, avg_val_accuracy = validation()
        print(str(epoch)+":"+"Avg Train Loss is :{:.6f}, Avg Train Accuracy is:{:.6f}%, Avg Test Loss is:{:.6f}, Avg Test Accuracy is:{:.6f}%".format(avg_train_loss, avg_train_acc, avg_val_loss, avg_val_accuracy)+"\n")
        with open(os.path.join(work_dir,"log.txt"), "a+") as f:
            f.write(str(epoch)+":"+"Avg Train Loss is :{:.6f}, Avg Train Accuracy is:{:.6f}%, Avg Test Loss is:{:.6f}, Avg Test Accuracy is:{:.6f}%".format(avg_train_loss, avg_train_acc, avg_val_loss, avg_val_accuracy)+"\n")
        if epoch%49==0 and epoch!=0:
            torch.save(model.state_dict(), os.path.join(model_save_path,"checkpoint_"+str(epoch).zfill(4)+".pkl"))
        if best_val_acc < avg_val_accuracy:
            best_epoch = epoch
            best_val_acc = avg_val_accuracy
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_checkpoint.pkl"))


    print("best_val_acc is :{:.6f}%".format(best_val_acc) + ", the epoch is :"+ str(best_epoch))
    with open(os.path.join(work_dir,"log.txt"), "a+") as f:
        f.write("best_val_acc is :{:.6f}%".format(best_val_acc) + ", the epoch is :"+ str(best_epoch)+"\n")


if __name__ == "__main__":
    print("\nStart training with noisy labels...")
    train()
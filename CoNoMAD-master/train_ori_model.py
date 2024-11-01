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


task_name = config.data.dataset_name + "_" + config.data.data_name +"_noise"+str(args.noise_ratio)\
    +"_model"+config.model.model_name + "_bs"+str(batch_size) + "_epochs50"+"_seed"+str(args.random_seed)\
    +str(args.order)+str(args.order_type)

print(task_name)

work_dir = os.path.join(os.path.dirname(__file__),"ori_deep", "record", config.data.dataset_name, config.data.file_name, task_name)


model_save_path = os.path.join(work_dir, "checkpoints")
os.makedirs(model_save_path, exist_ok=True)
model = net.get_model(config.model.model_name, num_classes=len(config.data.target_names), input_dim=config.data.data_dim)
model.to(device)
train_dataset, val_dataset = data.get_dataset(config=config.data, noise_ratio=args.noise_ratio, order=args.order, order_type=args.order_type)
train_loader= data.get_dataLoader(config.data, train_dataset)
val_loader= data.get_dataLoader(config.data, val_dataset)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=config.train.learning_rate, weight_decay=0.0005)


writer = SummaryWriter(os.path.join(work_dir,"Summary"))
global global_step
global_step = 0

def train():
    def train_one_epoch():
        global global_step
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        bar = tqdm(train_loader)
        for i,(X_train, y_train, cluster_labels) in enumerate(bar):
            X_train, y_train = X_train.to(device), y_train.to(device)
            cluster_labels = cluster_labels.to(device)
            outputs, outputs2 = model(X_train)
            _, pred = torch.max(outputs.data, 1)
            loss = loss_func(outputs, y_train)
            # loss2 = loss_func(outputs2, cluster_labels)
            # loss = loss + 0.3*loss2           
            # print(i, loss)
            # loss = loss/accumulation_steps   # Optional, if the losses are to be averaged over the training samples
            loss.backward()
            train_loss += loss.item()
            train_correct += torch.sum(pred == y_train.data)
            if((i+1)%accumulation_steps)==0:
                optimizer.step()
                optimizer.zero_grad()
                train_acc = train_correct/((i+1)*batch_size)
                train_loss_ = train_loss/((i+1)*batch_size)
                bar.set_description(f'Batch {i}/{len(train_loader)}')
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
                _, pred = torch.max(outputs.data, 1) # Returns the element with the largest value in each row, and returns its index
                val_correct += torch.sum(pred == y_val.data)
                loss = loss_func(outputs, y_val)
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
        # if epoch%10==0:
        #     torch.save(model.state_dict(), os.path.join(model_save_path,"checkpoint_"+str(epoch).zfill(4)+".pkl"))
        if best_val_acc < avg_val_accuracy:
            best_epoch = epoch
            best_val_acc = avg_val_accuracy
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_checkpoint.pkl"))
    print("best_val_acc is :{:.6f}%".format(best_val_acc) + ", the epoch is :"+ str(best_epoch))
    with open(os.path.join(work_dir,"log.txt"), "a+") as f:
        f.write("best_val_acc is :{:.6f}%".format(best_val_acc) + ", the epoch is :"+ str(best_epoch)+"\n")

if __name__ == "__main__":
    print("\nstart tarining....")
    train()

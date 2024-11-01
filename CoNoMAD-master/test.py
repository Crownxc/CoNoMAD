import argparse
import os
from sklearn import metrics
import torch
from tqdm import tqdm
import yaml
import net
import data
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

'''
python test.py --config config/drebin.yaml --testdir H:/09SE/Tools/7Drebin/data
'''
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
parser.add_argument('--lp',
                    # default="best_checkpoint.pkl",
                    default="checkpoint_0049.pkl",
                    type=str,
                    required=False,
                    help='Loaded models')
parser.add_argument('--testdir',
                    default="H:/09SE/Tools/7Drebin/data",
                    type=str,
                    metavar='PATH',
                    required=False,
                    help='Test Sample Path Prefix')
parser.add_argument('--txtpath',
                    default="../data_split/CIC2020_8076_label1/test.txt",
                    # default = r"H:\2304noise\data_split\mixed\drebin\test.txt",
                    type=str,
                    metavar='PATH',
                    required=False,
                    help='Path to test sample txt')
parser.add_argument('--noise_ratio', default=0.45, type=float, required=False)

parser.add_argument('--random_seed', default=42, type=int, required=False)
parser.add_argument('--order', default=False, type=bool, required=False, help='训练集样本顺序')
parser.add_argument('--order_type', default=0, type=int, required=False, help='样本顺序类型')
parser.add_argument('--alpha', default=0.4, type=float, required=False)
parser.add_argument('--beta', default=0.5, type=float, required=False)


args = parser.parse_args()
with open(args.config,"r",encoding="utf-8") as f:
    config = edict(yaml.load(f, Loader=yaml.FullLoader))
print(config.data.batch_size)

# GPU select
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = config.train.num_epochs
accumulation_steps = config.train.accumulation_steps
batch_size = config.train.batch_size


task_name = config.data.dataset_name + "_" + config.data.data_name +"_noise"+str(args.noise_ratio)\
    +"_model"+config.model.model_name + "_bs"+str(batch_size) + "_epochs"+ str(num_epochs)+"_seed"+str(args.random_seed)\
    +str(args.order)+str(args.order_type)+str(args.alpha)


print(task_name)

# work_dir = os.path.join(os.path.dirname(__file__), "record", config.data.dataset_name, config.data.data_name, task_name)
work_dir = os.path.join(os.path.dirname(__file__), "record", config.data.dataset_name, config.data.file_name, task_name)
model_save_path = os.path.join(work_dir, "checkpoints")


# model_load_path = model_save_path+"/checkpoint_"+str(args.lp).zfill(4)+".pkl"
model_load_path = os.path.join(model_save_path, args.lp)
model = net.get_model(config.model.model_name, num_classes=len(config.data.target_names), input_dim=config.data.data_dim, model_load_path=model_load_path)
model.to(device)
test_dataset = data.get_dataset(config.data, test_dir=args.testdir, test_txt=args.txtpath)
test_loader= data.get_dataLoader(config.data, test_dataset)


def test():
    model.eval()
    val_correct = 0
    # global global_step
    with torch.no_grad():
        test_y =[]
        test_result_y = []
        for i, (X_val, y_val) in enumerate(test_loader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs, _ = model(X_val)
            _, pred = torch.max(outputs.data, 1)
            val_correct += torch.sum(pred == y_val.data)
            test_y.extend(y_val.data.cpu())
            test_result_y.extend(pred.cpu())

        val_accuracy = 100. * float(val_correct) / float(len(test_dataset))

        print('Accuracy: {}/{} ({:.6f}%)\n'.format(val_correct, len(test_dataset), val_accuracy))
        test_report = metrics.classification_report(test_y, test_result_y, target_names=test_dataset.target_names, digits=6)

        return test_report

if __name__ == "__main__":
    test_report = test()
    print(test_report)
    save_path = "Report/RQ1_CIC_result/"+str(config.data.dataset_name)+"/"+str(config.data.file_name)
    # save_path = "../Report/mixed/" + "/" + str(config.data.data_name)

    os.makedirs(save_path, exist_ok=True)

    with open(save_path+"/"+"test"+"_noise"+str(args.noise_ratio)+".txt",'w') as f:
        f.write(str(test_report))

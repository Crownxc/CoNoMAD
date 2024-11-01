import os
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from torch.utils.data import Dataset

class Data_Mamafamily(Dataset):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:
        feature_name = "family_call_prob.npz"
        dataset_x = []
        dataset_y = []
        for data_base_pair in tqdm(data_base_list):
            data_path = os.path.join(data_dir, data_base_pair[0])
            apk_allFuncs_file = os.path.join(data_path, feature_name)
            if not os.path.exists(apk_allFuncs_file):
                continue
            feature = sp.load_npz(apk_allFuncs_file).todense().A.reshape(121, )
            label = data_base_pair[1]
            dataset_x.append(feature)
            dataset_y.append(label)
        self.data = np.array(dataset_x).astype(np.float32)
        self.label = dataset_y
        self.target_names = target_names

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
    
class Data_Mamapackage(Dataset):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:
        feature_name = "package_call_prob.npz"
        dataset_x = []
        dataset_y = []
        for data_base_pair in tqdm(data_base_list):
            data_path = os.path.join(data_dir, data_base_pair[0])
            apk_allFuncs_file = os.path.join(data_path, feature_name)
            if not os.path.exists(apk_allFuncs_file):
                continue
            feature = sp.load_npz(apk_allFuncs_file).todense().A.reshape(386*386, )
            label = data_base_pair[1]
            dataset_x.append(feature)
            dataset_y.append(label)
        self.data = np.array(dataset_x).astype(np.float32)
        self.label = dataset_y
        self.target_names = target_names

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
    
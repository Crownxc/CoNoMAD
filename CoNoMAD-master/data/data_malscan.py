import os
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer as CountV
import dill
import joblib
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import numpy as np
import faiss


class Data_Malscan(Dataset):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:
        feature_name = "malscan_center.npy"

        dataset_x = []
        dataset_y = []
        for data_base_pair in tqdm(data_base_list):
            data_path = os.path.join(data_dir, data_base_pair[0])
            apk_allFuncs_file = os.path.join(data_path, feature_name)
            if not os.path.exists(apk_allFuncs_file):
                continue
            feature = np.load(apk_allFuncs_file)
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
    


class Data_Malscan_Train(Data_Malscan):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:

        super(Data_Malscan_Train, self).__init__(dataset_name, data_dir, data_base_list, target_names)

        feature_name = "malscan_center.npy"

        dataset_x = []
        dataset_y = []
        for data_base_pair in tqdm(data_base_list):
            data_path = os.path.join(data_dir, data_base_pair[0])
            apk_allFuncs_file = os.path.join(data_path, feature_name)
            if not os.path.exists(apk_allFuncs_file):
                continue
            feature = np.load(apk_allFuncs_file)
            label = data_base_pair[1]
            dataset_x.append(feature)
            dataset_y.append(label)
        self.data = np.array(dataset_x).astype(np.float32)
        self.label = dataset_y
        self.target_names = target_names


        print("Start clustering")

        ncentroids = 2
        niter = 100
        verbose = True
        d = self.data.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(self.data[:])
        
        D, I = kmeans.index.search(self.data, 1)
        self.cluster_label = I

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.cluster_label[index][0]

    def __len__(self):
        return len(self.data)
    

class Data_Malscan_Test(Data_Malscan):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:

        super(Data_Malscan_Test, self).__init__(dataset_name, data_dir, data_base_list, target_names)
        feature_name = "malscan_center.npy"

        dataset_x = []
        dataset_y = []
        for data_base_pair in tqdm(data_base_list):
            data_path = os.path.join(data_dir, data_base_pair[0])
            apk_allFuncs_file = os.path.join(data_path, feature_name)
            if not os.path.exists(apk_allFuncs_file):
                continue
            feature = np.load(apk_allFuncs_file)
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
    


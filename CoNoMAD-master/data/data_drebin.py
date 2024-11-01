import os
import time

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer as CountV
import dill
import joblib
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
import numpy as np
import faiss



class HierarchicalKMeans:
    def __init__(self, data, n_clusters, n_levels, n_iterations=100, verbose=True):
        self.data = data
        self.n_clusters = n_clusters
        self.n_levels = n_levels
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.cluster_labels = []

    def fit(self):
        current_data = self.data
        for level in range(self.n_levels):
            if self.verbose:
                print(f"Level {level + 1} of {self.n_levels}")

            # faiss KMeans
            d = current_data.shape[1]
            kmeans = faiss.Kmeans(d, self.n_clusters, niter=self.n_iterations, verbose=self.verbose)
            kmeans.train(current_data)

            centroids = kmeans.centroids

            # Assign cluster labels to the current data
            D, I = kmeans.index.search(current_data, 1)
            self.cluster_labels.append(I)

            # If this is the last layer, skip the next layer of preparation
            if level == self.n_levels - 1:
                break

            # Preparing the next layer of data
            new_data = []
            for i in range(self.n_clusters):
                cluster_points = current_data[I.flatten() == i]
                new_data.append(cluster_points)

            # Splice the next layer of data together
            current_data = np.vstack(new_data)

        if self.verbose:
            print("Finish clustering")



class Data_Drebin(Dataset):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:

        feature_name = "drebin_feature.data"

        dataset_x = []
        dataset_y = []
        for data_base_pair in tqdm(data_base_list):
            data_path = os.path.join(data_dir, data_base_pair[0],feature_name)
            if os.path.exists(data_path):
                label = data_base_pair[1]
                dataset_x.append(data_path)
                dataset_y.append(label)

        self.data = None
        self.cluster_label = []
        self.ori_data = dataset_x
        self.label = dataset_y
        self.target_names = target_names
        save_path = os.path.join(os.path.dirname(__file__), dataset_name)
        print("gggg",save_path)
        os.makedirs(save_path, exist_ok=True)
        self.FV_path = os.path.join(save_path, "drebin_fv.pkl")


    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.ori_data)
    

class Data_Drebin_Train(Data_Drebin):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:

        super(Data_Drebin_Train, self).__init__(dataset_name, data_dir, data_base_list, target_names)

        if os.path.exists(self.FV_path):
            with open(self.FV_path, "rb") as f:
                self.FeatureVectorizer = dill.load(f)

            assert isinstance(self.FeatureVectorizer, CountV)
        else:
            self.FeatureVectorizer = CountV(
                input='filename',
                tokenizer=lambda x: x.split('\n'),
                token_pattern=None, 
                binary=True
            )
        self.data = self.FeatureVectorizer.fit_transform(self.ori_data).toarray().astype(np.float32)

        with open(self.FV_path, "wb") as f:
            dill.dump(self.FeatureVectorizer,f)

        #k-means
        print("Start clustering")

        ncentroids = 2
        niter = 100
        verbose = True
        d = self.data.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(self.data[:])
        print("Finish clustering")
        D, I = kmeans.index.search(self.data, 1)
        self.cluster_label = I



    def __getitem__(self, index):
        return self.data[index], self.label[index], self.cluster_label[index][0]
        
class Data_Drebin_Test(Data_Drebin):
    def __init__(self, dataset_name, data_dir, data_base_list, target_names=["0", "1"]) -> None:

        super(Data_Drebin_Test, self).__init__(dataset_name, data_dir, data_base_list, target_names)

        if not os.path.exists(self.FV_path):

            raise FileNotFoundError(f"{self.FV_path} does not exist")
        with open(self.FV_path, "rb") as f:
            self.FeatureVectorizer = dill.load(f)

        assert isinstance(self.FeatureVectorizer, CountV)

        self.data = self.FeatureVectorizer.transform(self.ori_data).toarray().astype(np.float32)



import torch
import random

from . import data_malscan, data_mamadroid, data_drebin

train_dataset_dict = {
    "malscan": data_malscan.Data_Malscan_Train,
    "mamafamily": data_mamadroid.Data_Mamafamily,
    "mamapackage": data_mamadroid.Data_Mamapackage,
    "drebin": data_drebin.Data_Drebin_Train,
}
test_dataset_dict = {
    "malscan": data_malscan.Data_Malscan_Test,
    "mamafamily": data_mamadroid.Data_Mamafamily,
    "mamapackage": data_mamadroid.Data_Mamapackage,
    "drebin": data_drebin.Data_Drebin_Test,
}

def get_datalist(data_txt):
    print(data_txt)
    with open(data_txt,'r') as f2:
        a = f2.read()
        data = eval(a)
    return data

def generate_pseudo_data(data_list, pseudo_labels):
    # Update the labels in the data list to pseudo-labels,
    # ensuring that the number of pseudo-labels matches the number of training data
    print(len(pseudo_labels))
    print(len(data_list))
    assert len(pseudo_labels) == len(data_list), "The number of pseudo labels must match the number of training samples"

    for index, (data, _) in enumerate(data_list):
        data_list[index][1] = pseudo_labels[index]
    return data_list


def noise_dataset(data_list, noise_ratio):
    # Number of randomly modified positions
    n = int(len(data_list)*noise_ratio)

    index_list = random.sample(range(len(data_list)), n)

    for index in index_list:
        data_list[index][1] = 1-data_list[index][1]

    return data_list

def order_noise_dataset(data_list, noise_ratio, order_type):
    # Number of randomly modified positions
    n = int(len(data_list)*noise_ratio)

    index_list = random.sample(range(len(data_list)), n)

    rest_list = list(set(random.sample(range(len(data_list)), len(data_list))) - set(index_list))
    new_data_list = []

    if order_type == 0:
     #=0 Indicates that the clean sample is in front
        for index in rest_list:
            new_data_list.append(data_list[index])

        for index in index_list:
            data_list[index][1] = 1 - data_list[index][1]
            new_data_list.append(data_list[index])
    if order_type == 1:
        #=1 Indicates clean samples in the back
        for index in index_list:
            data_list[index][1] = 1 - data_list[index][1]
            new_data_list.append(data_list[index])

        for index in rest_list:
            new_data_list.append(data_list[index])
    return new_data_list

# def create_dataset_with_pseudo_labels(train_dataset, pseudo_labels):



def get_dataset(config, noise_ratio=None, test_dir=None, test_txt=None, order=False, order_type=0, pseudo_labels=None):
    print("loading dataset......")
    if test_txt:
        test_data = get_datalist(test_txt)
        return test_dataset_dict[config.data_name](config.dataset_name, test_dir, test_data, config.target_names)

    train_data = get_datalist(config.train_txt)
    val_data = get_datalist(config.val_txt)

    random.shuffle(train_data)

    if noise_ratio:
        train_data = noise_dataset(train_data, noise_ratio)
    
    if order:
        train_data = order_noise_dataset(train_data, noise_ratio, order_type)

    if pseudo_labels is not None:
        train_data = generate_pseudo_data(train_data, pseudo_labels)

    return train_dataset_dict[config.data_name](config.dataset_name, config.data_dir, train_data, config.target_names),\
            test_dataset_dict[config.data_name](config.dataset_name, config.data_dir, val_data, config.target_names)



def get_dataLoader(config, dataset):
    # if config.data_name=="efcg":
    #     collate_fn = data_graph.collate_graph
    # else:
    #     collate_fn = torch.utils.data.dataloader.default_collate
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    return data_loader  

import torch
from . import dnn

model_dict = {
    "DNN": dnn.DNN,
}

def get_model(model_name, num_classes, input_dim, model_load_path=None):
    model = model_dict[model_name](num_classes, input_dim)
    if model_load_path:
        checkpoint = torch.load(model_load_path, map_location='cuda')
        model.load_state_dict(checkpoint)
    return model
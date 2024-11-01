
import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, num_classes=2, input_dim=512, hidden_dim=256):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.fc4 = nn.Linear(hidden_dim, 2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        out1 = self.fc3(x)
        out2 = self.fc4(x)
        return out1, out2




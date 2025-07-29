import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from nvflare.apis.dxo import DXO, DXOType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.shareable import Shareable
from nvflare.apis.trainer import Trainer
from nvflare.apis.fl_context import FLContext
from nvflare.apis.utils.shareable_utils import convert_model_to_shareable, extract_model_from_shareable

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

class CNNTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        model_data = extract_model_from_shareable(shareable)
        model = CNN().to(self.device)
        model.load_state_dict(model_data)

        train_loader = DataLoader(
            datasets.FashionMNIST(
                "./data", train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor()])
            ),
            batch_size=32,
            shuffle=True,
        )

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(1):  # 1 epoch per round
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

        new_weights = model.state_dict()
        dxo = DXO(data_kind=DXOType.WEIGHTS, data=new_weights)
        return dxo.to_shareable()

    def abort(self, fl_ctx: FLContext):
        pass

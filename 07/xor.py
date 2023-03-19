import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch import nn

X_train = np.array([
    [0, 1],
    [1, 1],
    [1, 0],
    [1, 1],
    [0, 0],
    [1, 0]
], dtype=np.float32)
y_train = np.array([1, 0, 1, 0, 0, 1], dtype=np.float32)

X_test = np.array([
    [0, 0],
    [1, 0]
], dtype=np.float32)
y_test = np.array([0, 1], dtype=np.float32)


class XORData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class XORLinear(nn.Module):
    def __init__(self):
        super(XORLinear, self).__init__()
        self.layer = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x):
        return self.layer(x)


class XORNonLinear(nn.Module):
    def __init__(self):
        super(XORNonLinear, self).__init__()
        self.sequence = nn.Sequential(
            nn.Linear(in_features=2, out_features=2, bias=True),
            nn.Sigmoid(),
            nn.Linear(in_features=2, out_features=1, bias=True)
        )

    def forward(self, x):
        return self.sequence(x)


batch_size = 1

train_data = XORData(X_train, y_train)
train_dataloader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = XORData(X_test, y_test)
test_dataloader = DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=True)


def train_model(model, verbose=False):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    num_epochs = 1000
    plot_every = 50
    current_loss = 0
    total_loss = []

    for epoch in range(num_epochs + 1):
        for X, y in train_dataloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(X)
            loss = loss_fn(pred, y.unsqueeze(-1))
            current_loss += loss.item()
            loss.backward()

            optimizer.step()

        if epoch % plot_every == 0:
            total_loss.append(current_loss / plot_every)
            current_loss = 0

        if epoch % 250 == 0:
            print(f'Epoch: {epoch} completed')

    if verbose:
        epochs = np.linspace(0, num_epochs, len(total_loss))
        plt.plot(epochs, total_loss)
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.show()

    return model


def evaluate_model(model, verbose=False):
    total = 0
    correct = 0
    y_pred = []
    y_test = []

    with torch.no_grad():
        for X, y in test_dataloader:
            outputs = model(X)
            predicted = np.where(outputs < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            y_pred.append(predicted)
            y_test.append(y)
            total += y.size(0)
            correct += (predicted == y.numpy()).sum().item()

    accuracy = 100 * correct // total

    if verbose:
        print("Printing results...")
        print(f'Accuracy of the network: {accuracy}%')

    return accuracy


xor_nonlinear_model = XORNonLinear()
xor_nonlinear_model = train_model(xor_nonlinear_model, verbose=True)
accuracy = evaluate_model(xor_nonlinear_model, verbose=True)

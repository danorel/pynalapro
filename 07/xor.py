import itertools
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn

X = np.array([
    [0, 1],
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1]
], dtype=np.float32)

y = np.array([1, 0, 0, 1, 1], dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=26)


class XORData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class XORNeuralNetwork(nn.Module):
    def __init__(self):
        super(XORNeuralNetwork, self).__init__()
        self.layer = nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x):
        return self.layer(x)


batch_size = 1

train_data = XORData(X_train, y_train)
train_dataloader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = XORData(X_test, y_test)
test_dataloader = DataLoader(
    dataset=test_data, batch_size=batch_size, shuffle=True)

input_dim = 1
output_dim = 1

model = XORNeuralNetwork()

learning_rate = 0.1
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 10
loss_values = []

for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

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

print(
    f'Accuracy of the network on the 1 test instances: {100 * correct // total}%')

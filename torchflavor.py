
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import glob



csv_path = r"A:\ML_start\pointcloud_proj\raw\csv\*.csv"
txt_path = r"A:\ML_start\pointcloud_proj\raw\txt\*.txt"
csv_files = sorted(glob.glob(csv_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
txt_files = sorted(glob.glob(txt_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
# Define the PyTorch model:
class ScatterplotModel(nn.Module):
    def __init__(self):
        super(ScatterplotModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#Load the training data from the CSV files:
data = []
for file in csv_files:
    df = pd.read_csv(os.path.join(csv_dir, file))
    data.append(df.values)

data = np.concatenate(data)


#Load the ground truth vectors from the TXT files:
labels = []
for file in txt_files:
    with open(os.path.join(txt_dir, file), 'r') as f:
        label = list(map(float, f.read().split(',')))
        labels.append(label)

labels = np.array(labels)


# Normalize the data and labels:
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
labels_mean = np.mean(labels, axis=0)
labels_std = np.std(labels, axis=0)

data = (data - data_mean) / data_std
labels = (labels - labels_mean) / labels_std


# Convert the data and labels to PyTorch tensors:

data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)


# Initialize the model and define the loss function and optimizer:
model = ScatterplotModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Train the model:
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Predict the seed vector using the trained model:
test_data = torch.tensor([[x1, y1], [x2, y2], [x3, y3]], dtype=torch.float32)  # Replace with your test data

model.eval()
with torch.no_grad():
    predicted_labels = model(test_data)

predicted_labels = predicted_labels * labels_std + labels_mean


#replace `x1, y1, x2, y2, x3, y3` with the actual values of the scatterplot points you want to predict the seed vector for.

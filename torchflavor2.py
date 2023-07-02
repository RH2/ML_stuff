import glob
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

# Step 0: Specify Dataset Directories and methods for importing the data
csv_path = r"A:\ML_start\pointcloud_proj\raw\csv\*.csv"
txt_path = r"A:\ML_start\pointcloud_proj\raw\txt\*.txt"
csv_files = sorted(glob.glob(csv_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
txt_files = sorted(glob.glob(txt_path), key=lambda x: int(''.join(filter(str.isdigit, x))))

def load_csv_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=True)
    # Extract the x, y, z, r, g, and b attributes
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    r = data[:, 3]
    g = data[:, 4]
    b = data[:, 5]

    # print("values:",x,y,z,r,g,b)
    # coordinates = [float(val) for val in line.split(",")]
    # return position_numpy_vector, color_numpy_vector
    return x,y,z,r,g,b

def load_txt_data(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        coordinates = [float(val) for val in line.split(",")]
    return np.array(coordinates)



X_train = []
y_train = []

for csv_file, txt_file in zip(csv_files, txt_files):
    # P, Cd = load_csv_data(csv_file)
    # P = np.array(P, dtype=np.float32)
    # Cd = np.array(Cd, dtype=np.float32)
    # X_train.append(np.column_stack((P, Cd)))
    x,y,z,r,g,b = load_csv_data(csv_file)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    z = np.array(z, dtype=np.float32)
    r = np.array(r, dtype=np.float32)
    g = np.array(g, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    X_train.append(np.column_stack((x,y,z,r,g,b)))

    
    seed_vector = load_txt_data(txt_file)
    y_train.append(seed_vector)

X_train = np.array(X_train)
y_train = np.array(y_train)


# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)



# Step 1: Define the neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # Assuming P and Cd are concatenated as input
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Assuming the ground truth vector has 3 dimensions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 2: Prepare the data
X_train = torch.Tensor(X_train)  # Convert the numpy arrays to PyTorch tensors
y_train = torch.Tensor(y_train)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 3: Initialize the model, loss function, and optimizer
model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Step 5: Evaluate the model on the validation set
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val)
    print(f"Validation Loss: {val_loss.item():.4f}")

# Step 6: Use the trained model for prediction
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)  # Replace X_test with your test data

# test_outputs will contain the predicted vectors for the Cd color and P position

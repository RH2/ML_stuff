import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
import numpy as np

# Step 0: Specify Dataset Directories and methods for importing the data
csv_path = r"A:\ML_start\pointcloud_proj\raw\csv\*.csv"
txt_path = r"A:\ML_start\pointcloud_proj\raw\txt\*.txt"

def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    pvals= data["P"].values
    cvals= data["Cd"].values
    print("csv data", pvals[0])

    # Remove parentheses and split the string into individual values
    pSplits = pvals.split(',')
    # Convert string values to floating-point numbers
    position_values = [float(value) for value in pSplits]
    # Create a NumPy array from the float values
    position_numpy_vector = np.array(position_values)

    #Do the same for colors

    cSplits = cvals.split(",")
    color_values = [float(value) for value in cSplits]
    color_numpy_vector = np.array(color_values)





    # coordinates = [float(val) for val in line.split(",")]
    return position_numpy_vector, color_numpy_vector

def load_txt_data(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        coordinates = [float(val) for val in line.split(",")]
    return np.array(coordinates)

csv_files = sorted(glob.glob(csv_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
txt_files = sorted(glob.glob(txt_path), key=lambda x: int(''.join(filter(str.isdigit, x))))

X_train = []
y_train = []

for csv_file, txt_file in zip(csv_files, txt_files):
    P, Cd = load_csv_data(csv_file)
    P = np.array(P, dtype=np.float32)
    Cd = np.array(Cd, dtype=np.float32)
    X_train.append(np.column_stack((P, Cd)))
    
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

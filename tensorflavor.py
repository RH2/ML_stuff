import tensorflow as tf
import pandas as pd
import numpy as np
import glob

csv_path = r"A:\ML_start\pointcloud_proj\raw\csv\*.csv"
txt_path = r"A:\ML_start\pointcloud_proj\raw\txt\*.txt"

def load_csv_data(file_path):
    data = pd.read_csv(file_path)
    return data["P"].values, data["Cd"].values

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
    seed_vector = load_txt_data(txt_file)
    X_train.append(np.column_stack((P, Cd)))
    y_train.append(seed_vector)

X_train = np.array(X_train)
y_train = np.array(y_train)




model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32)
predictions = model.predict(X_test)

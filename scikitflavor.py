# MY GOAL IS TO BE SOMEONE WHO CAN DO THIS STUFF, SO LETS GO!

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import svm
import os
import glob

#PART 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("BEGIN...")
# Specify the directory path and file extension
directory_path = r'A:\ML_start\pointcloud_proj\raw\csv\*.csv'
file_extension = '.csv'  # Replace with the file extension you want to load
file_paths = glob.glob(directory_path)

for file in file_paths:
    print(file)
#data = pd.read_csv(r'')



# Load all CSV files into a single frame
combined_data = pd.DataFrame()
sorted_file_names = sorted(file_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
for file_path in sorted_file_names:
    data = pd.read_csv(file_path)
    combined_data = combined_data.append(data, ignore_index=True)

attributes = combined_data[['P',"Cd"]]
print(attributes)

#Load the goal attribute...
goal_combined = []
goal_paths = glob.glob(r'A:\ML_start\pointcloud_proj\raw\txt\*.txt')
#Sequence is *very* important
sorted_file_names = sorted(goal_paths, key=lambda x: int(''.join(filter(str.isdigit, x))))
for file in sorted_file_names:
    goal_data = pd.read_csv(file, sep=",", header=None)
    goal_data.columns = ["x","y","z"]
    goal_combined.append(goal_data)

print(goal_combined)

#Split the data into training and testing sets
#Part 1: the goals!
split_index = len(goal_combined) // 2
print("SPLIT_INDEX:", split_index)
goal_train = goal_combined[:split_index]
goal_evaluate = goal_combined[split_index:]
print("train:",goal_train)
print("evaluate:",goal_evaluate)

#Part 2: the data!
data_train = attributes[:split_index]
data_evaluate = attributes[:split_index:]


#Initialize a classifier & train
classifier = DecisionTreeClassifier()
classifier.fit(data_train,goal_train)

#Make predictions using the model
predictions = classifier.predict(data_evaluate)

#Evaluate Performance
score = classifier.score(predictions,goal_evaluate) 
print("Score:", score)







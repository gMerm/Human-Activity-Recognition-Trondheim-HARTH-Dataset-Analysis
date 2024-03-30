import os
import pandas as pd

#Default path to the dataset
default_path = os.getcwd() + "/harth/"

#Label dictionary: Connects the Label number to the activity
label_dict = {
    1: "Walking",
    2: "Running",
    3: "Shuffling",
    4: "Stairs (Ascending)",
    5: "Stairs (Descending)",
    6: "Standing",
    7: "Sitting",
    8: "Lying",
    13: "Cycling (sit)",
    14: "Cycling (Stand)",
    130: "Cycling (sit, inactive)",
    140: "Cycling (stand, inactive)"
}


# This Function lists all the files in the directory
def list_files(path=default_path):
    if(path is None):
        path = os.getcwd() + "/harth"
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)
    return files

# This Function reads the files and returns the data
def read_files(files,path=default_path):
    data = []
    for file in files:
        file = path + file
        data.append(pd.read_csv(file))
    return data

# This Function returns the showing percentage of the labels
def get_label_stats(frame):
    label_count = frame['label'].value_counts()
    percentages = label_count*100 / len(frame)
    print("Percentages of each label:")
    print(percentages)

def drop_columns(data):
    for i in range(len(data)):
        data[i]=data[i].dropna()
        data[i]=data[i].drop(columns=['timestamp'])
        if "Unnamed: 0" in data[i].columns:
            data[i]=data[i].drop(columns=['Unnamed: 0'])
        if "index" in data[i].columns:
            data[i]=data[i].drop(columns=['index'])
    return data
    
    
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from init_funcs import *

print("Neural Network Classifier\n")
print("Loading Data\n")
files = list_files()
files.sort()
print("Total Number of files:", len(files))
data = read_files(files)
print("\nData Loaded")

accuracy=[]
precision=[]
recall=[]
f1=[]
confusion_matrix=[]
classification_report=[]

#Preprocessing and deleting unecessary columns
for i in range(len(data)):
    data[i]=data[i].dropna()
    data[i]=data[i].drop(columns=['timestamp'])
    if "Unnamed: 0" in data[i].columns:
        data[i]=data[i].drop(columns=['Unnamed: 0'])
    if "index" in data[i].columns:
        data[i]=data[i].drop(columns=['index'])


for i in range(len(data)):
    local_accuracy = 0
    local_precision = 0
    local_recall = 0
    local_f1 = 0

    print(get_label_stats(data[i]))
    print("NN Model for File ", files[i])
    print("Training Model")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data[i].iloc[:,0:len(data[i].columns)])
    Y_train = data[i]['label']
    nn_model = MLPClassifier(hidden_layer_sizes=(30, 20, 10), max_iter=250)
    nn_model.fit(X_train, Y_train)
    print("Model Trained\n")

    for j in range(len(data)):
        if i != j:
            
            X_test = scaler.transform(data[j].iloc[:,0:len(data[j].columns)])
            Y_test = data[j]['label']
            Y_pred = nn_model.predict(X_test)

            local_accuracy += accuracy_score(Y_test, Y_pred)
            local_precision += precision_score(Y_test, Y_pred, average='macro',zero_division=0)
            local_recall += recall_score(Y_test, Y_pred, average='macro',zero_division=0)
            local_f1 += f1_score(Y_test, Y_pred, average='macro',zero_division=0)
            #confusion_matrix.append(confusion_matrix(Y_test, Y_pred))
            #classification_report.append(classification_report(Y_test, Y_pred))

            print("\nFile ", files[i], " vs File ", files[j])
            print("Accuracy: ", accuracy_score(Y_test, Y_pred))
            print("Precision: ", precision_score(Y_test, Y_pred, average='macro',zero_division=0))
            print("Recall: ", recall_score(Y_test, Y_pred, average='macro',zero_division=0))
            print("F1 Score: ", f1_score(Y_test, Y_pred, average='macro',zero_division=0))
            #print("Confusion Matrix: ", confusion_matrix(Y_test, Y_pred))
            #print("Classification Report: ", classification_report(Y_test, Y_pred))
            print("Testing Data Statistics\n")
            print(get_label_stats(data[j]))
            print("\n") 
    accuracy.append(local_accuracy/(len(data)-1))
    precision.append(local_precision/(len(data)-1))
    recall.append(local_recall/(len(data)-1))
    f1.append(local_f1/(len(data)-1))

for i in range(len(data)):
    print("File ", files[i])
    print("Accuracy: ", accuracy[i])
    print("Precision: ", precision[i])
    print("Recall: ", recall[i])
    print("F1 Score: ", f1[i])
    print("\n")
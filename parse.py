import pandas as pd

#dictionary to store activity codes and their respective top 2 features
activity_features = {}

#open file to read
with open('activity_code_results.txt', 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        #remove whitespaces
        line = lines[i].strip()  
        if line.startswith("Activity code:"):
            #extract the activity code
            activity_code = int(line.split(":")[1].strip()) 

            #create an empty list to store top 2 features
            top_features = []  

            #extract top 2 features
            #start searching from the line after "Top features affecting activity code:"
            j = i + 3  
            count = 0
            while j < len(lines) and lines[j].strip() != "-" * 50 and count < 3:
                #extract the feature name (the part before the space)
                feature_name = lines[j].strip().split()[0]
                top_features.append(feature_name)
                j += 1  
                count += 1

            #pad the top features list with empty strings if it has less than 2 features
            while len(top_features) < 3:
                top_features.append('')

            #add activity code and its top 2 features to the dictionary
            activity_features[activity_code] = top_features 

            #move to the next activity code section
            i = j  
        else:
            i += 1  

#convert the dictionary to a pandas DataFrame for easier manipulation
activity_features_df = pd.DataFrame(activity_features)
activity_features_df = activity_features_df.transpose()

#disp
print(activity_features_df)

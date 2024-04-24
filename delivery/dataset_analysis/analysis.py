import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
import sys
import numpy as np


#ask user if he wants to run the program
def ask(message):
    while True:
        user_input = input(f"{message} (y/n): ").strip().lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

if len(sys.argv) > 1 and sys.argv[1] == '-p' and ask("Do you want to run the program?"):

    #read data
    file_paths = ["harth/S006.csv", "harth/S008.csv", "harth/S009.csv", "harth/S010.csv", "harth/S012.csv", "harth/S013.csv", "harth/S014.csv", "harth/S015.csv", "harth/S016.csv", "harth/S017.csv", "harth/S018.csv", "harth/S019.csv", "harth/S020.csv", "harth/S021.csv", "harth/S022.csv", "harth/S023.csv", "harth/S024.csv", "harth/S025.csv", "harth/S026.csv", "harth/S027.csv", "harth/S028.csv", "harth/S029.csv"]
    #file_paths = ["harth/S006.csv"]
    dfs = []

    #exclude collunns from .csv files (files 15,21,23 include 1 collumn that isn't needed)
    for file in file_paths:
        #read .csv file
        df = pd.read_csv(file)
        
        #exclude unwanted columns if they exist in the .csv file
        if 'index' in df.columns:
            df.drop(columns=['index'], inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        
        #append df to list
        dfs.append(df)


    concatenated_df = pd.concat(dfs)

    # Sample the data to speed up the visualization
    sample_size = 10000
    random_indices = np.random.choice(concatenated_df.shape[0], sample_size, replace=False)
    sampled_df = concatenated_df.iloc[random_indices]

    #pairplot to visualize relationships between features & labels
    features_subset1 = ['back_x', 'back_y', 'back_z']
    features_subset2 = ['thigh_x', 'thigh_y', 'thigh_z']

    sns.pairplot(sampled_df, hue='label', markers='.', vars=features_subset1, palette='Set2')
    plt.suptitle('Pair Plot - Features Subset 1', y=1.02)
    plt.show()

    sns.pairplot(sampled_df, hue='label', markers='.', vars=features_subset2, palette='Set2')
    plt.suptitle('Pair Plot - Features Subset 2', y=1.02)
    plt.show()

    #scatterplot plots for feature - label
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    sns.scatterplot(data=sampled_df, x='back_x', y='label', ax=axes[0][0])
    axes[0][1].set_title('back_x')

    
    sns.scatterplot(data=sampled_df, x='back_y', y='label', ax=axes[0][1])
    axes[0][1].set_title('back_y')

    sns.scatterplot(data=sampled_df, x='back_z', y='label', ax=axes[0][2])
    axes[0][2].set_title('back_z')

    sns.scatterplot(data=sampled_df, x='thigh_x', y='label', ax=axes[1][0])
    axes[1][1].set_title('thigh_x')

    
    sns.scatterplot(data=sampled_df, x='thigh_y', y='label', ax=axes[1][1])
    axes[1][1].set_title('thigh_y')

    sns.scatterplot(data=sampled_df, x='thigh_z', y='label', ax=axes[1][2])
    axes[1][2].set_title('thigh_z')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    features = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

    for i, feature in enumerate(features):
        sns.violinplot(data=sampled_df, x='label', y=feature, ax=axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(feature)

    plt.tight_layout()
    plt.show()

    print('eftasa edw')

    # Descriptive analysis
    #count = number of non-null values in each coll
    #mean = average value of each coll
    #std = standard deviation, which measures the spread of values around the mean
    #min: minimum value in each column
    #25%: first quartile (25th percentile), which separates the lowest 25% of the data from the rest
    #50%: second quartile (median or 50th percentile), which represents the middle value of the dataset
    #75%: third quartile (75th percentile), which separates the lowest 75% of the data from the rest
    #max: maximum value in each coll
        
    #if file already exists drop it
    '''
    if os.path.exists("activity_code_results.txt"):
        os.remove("activity_code_results.txt")
    '''

    with open('activity_code_results.txt', 'w') as f:
        for i, df in enumerate(dfs):

            f.write(f"Descriptive & Statistic analysis for file: {file_paths[i]}\n\n")
            f.write(f"{df.describe()}\n\n")
            f.write(f"{df.groupby('label').agg({'back_x': ['mean', 'std'], 'back_y': ['mean', 'std'], 'back_z': ['mean', 'std'], 'thigh_x': ['mean', 'std'], 'thigh_y': ['mean', 'std'], 'thigh_z': ['mean', 'std']})}\n\n")

            #remove timestamp collumn in order not to have errors in the correlation matrices
            if 'timestamp' in df.columns:
                df.drop(columns=['timestamp'], inplace=True)

            #get the activities code from feature named "label" from .csv files in order to represent them in the matrices
            activity_codes = df['label'].unique()


            #subplots
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            #correlation between accelerometer readings
            correlation_matrix = df.iloc[:, 1:].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=activity_codes, yticklabels=activity_codes, ax=axes[0])
            axes[0].set_title(f"Accelerometer Readings")

            #correlations between mean accelerometer readings and activity code
            mean_correlation = df.groupby('label').mean().corr()
            sns.heatmap(mean_correlation, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=activity_codes, yticklabels=activity_codes, ax=axes[1])
            axes[1].set_title(f"Mean Accelerometer Readings by Activity")

            #correlations between standard deviation of accelerometer readings and activity code
            std_correlation = df.groupby('label').std().corr()
            sns.heatmap(std_correlation, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=activity_codes, yticklabels=activity_codes, ax=axes[2])
            axes[2].set_title(f"Standard Deviation of Accelerometer Readings by Activity")

            plt.suptitle(f"File: {file_paths[i]}")
            plt.tight_layout()
            plt.show()


            #correlations between accelerometer readings and each activity code
            for activity_code in activity_codes:
                #filter data for current activity code
                activity_df = df[df['label'] == activity_code]

                #correlations between accelerometer readings and activity code
                correlation_matrix = df.corrwith(df['label']).dropna().sort_values(ascending=False)
            
                #print current activity code and top features affecting it
                #used iloc[1:6] to skip the first feature that comes up which is "label" hahaha
                f.write(f"Activity code: {activity_code}\n")
                f.write("Top features affecting activity code:\n")
                f.write(f"{correlation_matrix.iloc[1:6]}\n")
                f.write("-" * 50 + "\n")


            #time.sleep(3);
                








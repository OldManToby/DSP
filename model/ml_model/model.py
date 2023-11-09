#Toby Warn (20026345)

#Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

def load_csv(folder_path):
    all_data = []
    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                data = pd.read_csv(file_path)
                all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

folder_path = r'C:\Users\Toby\Documents\GitHub\DSP\team_stats'

data = load_csv(folder_path)
print(data)


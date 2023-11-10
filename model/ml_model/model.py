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
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

#load in the data through subfolders within 'team_stats'
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
print(data.isnull().sum())

# Handle missing values by filling with the mean of each column
data.fillna(data.mean(), inplace=True)

# Drop rows containing any remaining missing values
data.dropna(inplace=True)

# Encoding categorical variables (team names)
label_encoder = LabelEncoder()
data['Team'] = label_encoder.fit_transform(data['Team'])

# Feature Scaling
scaler = StandardScaler()
numerical_features = ['Att', 'Cmp', 'Cmp.1', 'Yds/Att', 'Yds', 'TD', 'INT', '1st', '1st%', 'Sck', 'Rush', 'YPC', 'Rush.1', 'Rush.2', 'FR', 'SFTY', '3rd', '3rd.1', '4th', '4th.1', 'Scrm', 'FF', 'FR.1', 'INT.1', 'INT.2', 'Lng', 'Pass', 'Rate', '20+', '40+', 'SckY', 'Rush.3', 'Rec', 'Yds/Rec', 'Rec.1', 'Rec.2', 'Rec.3', 'Rsh', 'Tot', '2-PT', 'FGM', 'FG', '1-19 >', '20-29 >', '30-39 >', '40-49 >', '50-59 >', '60+ >', 'FG.1', 'XPM', 'XP', 'KRet', 'PRet', 'KO', 'TB', 'TB.1', 'Ret', 'Ret.1', 'OSK', 'OSK.1', 'OOB', 'Avg', 'FC', 'FUM']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Check for Null Values
print("Null Values Check:")
print(data.isnull().sum())

# Check Encoded Categorical Values
print("\nUnique Values in 'Team' after Encoding:")
print(data['Team'].unique())

# Check Scaled Numerical Features
print("\nSummary Statistics of Scaled Numerical Features:")
print(data[numerical_features].describe())

# Correlation Analysis
correlation_matrix = data[numerical_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Visualizing Data Distributions with KDE and Boxplots
for feature in numerical_features:
    plt.figure(figsize=(12, 6))
    
    # Histogram with Kernel Density Estimation (KDE)
    sns.histplot(data[feature], kde=True, color='blue', bins=30)
    
    # Boxplot to identify outliers
    sns.boxplot(x=data[feature], color='red', width=0.2)
    
    plt.title(f'Distribution of {feature}')
    plt.show()

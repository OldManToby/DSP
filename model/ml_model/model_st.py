# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data from subfolders named 'special_team' within 'team_stats'
def load_csv_special_team(folder_path):
    all_data = []
    for subdir, _, files in os.walk(folder_path):
        # Add condition to filter subfolders named 'special_team'
        if os.path.basename(subdir).lower() == 'special_team':
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    data = pd.read_csv(file_path)
                    all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# Specify folder path for special_team subfolders
special_team_folder_path = r'C:\Users\Toby\Documents\GitHub\DSP\team_stats'

# Load data only from special_team subfolders
special_team_data = load_csv_special_team(special_team_folder_path)

# Display loaded data
print(special_team_data)
print(special_team_data.isnull().sum())

# Handle missing values by filling with the mean of each column
special_team_data.fillna(special_team_data.mean(), inplace=True)

# Drop rows containing any remaining missing values
special_team_data.dropna(inplace=True)

# Encoding categorical variables (team names)
label_encoder = LabelEncoder()
special_team_data['Team'] = label_encoder.fit_transform(special_team_data['Team'])

# Feature Scaling
scaler = StandardScaler()

# Drop constant columns 'Att' and 'Yds'
constant_columns = ['Att', 'Yds']
special_team_data = special_team_data.drop(columns=constant_columns)

# Update numerical features based on columns present in the special_team data
numerical_features = list(special_team_data.columns[special_team_data.dtypes == 'float64'])

# Exclude 'Team' from numerical features if it is in the list
if 'Team' in numerical_features:
    numerical_features.remove('Team')

# Feature Scaling
scaler = StandardScaler()
special_team_data[numerical_features] = scaler.fit_transform(special_team_data[numerical_features])

# Check for Null Values
print("Null Values Check:")
print(special_team_data.isnull().sum())

# Check Encoded Categorical Values
print("\nUnique Values in 'Team' after Encoding:")
print(special_team_data['Team'].unique())

# Check Scaled Numerical Features
print("\nSummary Statistics of Scaled Numerical Features:")
print(special_team_data[numerical_features].describe())

# Correlation Analysis
correlation_matrix = special_team_data[numerical_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Visualizing Data Distributions with KDE and Boxplots
for feature in numerical_features:
    plt.figure(figsize=(12, 6))

    # Histogram with Kernel Density Estimation (KDE)
    sns.histplot(special_team_data[feature], kde=True, color='blue', bins=30)

    # Boxplot to identify outliers
    sns.boxplot(x=special_team_data[feature], color='red', width=0.2)

    plt.title(f'Distribution of {feature}')
    plt.show()

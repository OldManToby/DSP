# Toby Warn (20026345)

# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data from subfolders named 'offensive' within 'team_stats'
def load_csv_offensive(folder_path):
    all_data = []
    for subdir, _, files in os.walk(folder_path):
        # Add condition to filter subfolders named 'offensive'
        if os.path.basename(subdir).lower() == 'offensive':
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    data = pd.read_csv(file_path)
                    all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# Specify folder path for offensive subfolders
offensive_folder_path = r'C:\Users\Toby\Documents\GitHub\DSP\team_stats'

# Load data only from offensive subfolders
offensive_data = load_csv_offensive(offensive_folder_path)

# Display loaded data
print(offensive_data)
print(offensive_data.isnull().sum())

# Handle missing values by filling with the mean of each column
offensive_data.fillna(offensive_data.mean(), inplace=True)

# Drop rows containing any remaining missing values
offensive_data.dropna(inplace=True)

# Encoding categorical variables (team names)
label_encoder = LabelEncoder()
offensive_data['Team'] = label_encoder.fit_transform(offensive_data['Team'])

# Feature Scaling
scaler = StandardScaler()

# Update numerical features based on columns present in the offensive data
numerical_features = list(offensive_data.columns[offensive_data.dtypes == 'float64'])

# Exclude 'Team' from numerical features if it is in the list
if 'Team' in numerical_features:
    numerical_features.remove('Team')

# Update the original code with the modified numerical_features list
offensive_data[numerical_features] = scaler.fit_transform(offensive_data[numerical_features])

# Update the original code with the modified numerical_features list
offensive_data[numerical_features] = scaler.fit_transform(offensive_data[numerical_features])

# Check for Null Values
print("Null Values Check:")
print(offensive_data.isnull().sum())

# Check Encoded Categorical Values
print("\nUnique Values in 'Team' after Encoding:")
print(offensive_data['Team'].unique())

# Check Scaled Numerical Features
print("\nSummary Statistics of Scaled Numerical Features:")
print(offensive_data[numerical_features].describe())

# Correlation Analysis
correlation_matrix = offensive_data[numerical_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Visualizing Data Distributions with KDE and Boxplots
for feature in numerical_features:
    plt.figure(figsize=(12, 6))

    # Histogram with Kernel Density Estimation (KDE)
    sns.histplot(offensive_data[feature], kde=True, color='blue', bins=30)

    # Boxplot to identify outliers
    sns.boxplot(x=offensive_data[feature], color='red', width=0.2)

    plt.title(f'Distribution of {feature}')
    plt.show()

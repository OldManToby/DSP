# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data from subfolders named 'defensive' within 'team_stats'
def load_csv_defensive(folder_path):
    all_data = []
    for subdir, _, files in os.walk(folder_path):
        # Add condition to filter subfolders named 'defensive'
        if os.path.basename(subdir).lower() == 'defensive':
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    data = pd.read_csv(file_path)
                    all_data.append(data)
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# Specify folder path for defensive subfolders
defensive_folder_path = r'C:\Users\Toby\Documents\GitHub\DSP\team_stats'

# Load data only from defensive subfolders
defensive_data = load_csv_defensive(defensive_folder_path)

# Display loaded data
print(defensive_data)
print(defensive_data.isnull().sum())

# Handle missing values by filling with the mean of each column
defensive_data.fillna(defensive_data.mean(), inplace=True)

# Drop rows containing any remaining missing values
defensive_data.dropna(inplace=True)

# Encoding categorical variables (team names)
label_encoder = LabelEncoder()
defensive_data['Team'] = label_encoder.fit_transform(defensive_data['Team'])

# Feature Scaling
scaler = StandardScaler()

# Drop constant columns 'Att' and 'Yds'
constant_columns = ['Att', 'Yds']
defensive_data = defensive_data.drop(columns=constant_columns)

# Update numerical features based on columns present in the defensive data
numerical_features = list(defensive_data.columns[defensive_data.dtypes == 'float64'])

# Exclude 'Team' from numerical features if it is in the list
if 'Team' in numerical_features:
    numerical_features.remove('Team')

# Feature Scaling
scaler = StandardScaler()
defensive_data[numerical_features] = scaler.fit_transform(defensive_data[numerical_features])

# Check for Null Values
print("Null Values Check:")
print(defensive_data.isnull().sum())

# Check Encoded Categorical Values
print("\nUnique Values in 'Team' after Encoding:")
print(defensive_data['Team'].unique())

# Check Scaled Numerical Features
print("\nSummary Statistics of Scaled Numerical Features:")
print(defensive_data[numerical_features].describe())

# Correlation Analysis
correlation_matrix = defensive_data[numerical_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Visualizing Data Distributions with KDE and Boxplots
for feature in numerical_features:
    plt.figure(figsize=(12, 6))

    # Histogram with Kernel Density Estimation (KDE)
    sns.histplot(defensive_data[feature], kde=True, color='blue', bins=30)

    # Boxplot to identify outliers
    sns.boxplot(x=defensive_data[feature], color='red', width=0.2)

    plt.title(f'Distribution of {feature}')
    plt.show()


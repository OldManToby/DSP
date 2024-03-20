import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function to load datasets
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Generalized data cleaning function
def clean_data(df, columns_to_drop, preserve_col=None):
    preserved_data = None
    if preserve_col and preserve_col in df.columns:
        preserved_data = df[preserve_col].reset_index(drop=True)
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    # Fill missing numeric values with the mean
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    if preserved_data is not None:
        df[preserve_col] = preserved_data
    return df

# Function to plot correlation matrix
def plot_correlation_matrix(df, title):
    # Exclude non-numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()  # Calculate correlation only on numeric columns
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.show()


# Paths to the datasets
base_directory = './clean_team_stats/'  # Adjust this if your datasets are not in the current directory
d_merged_data_path = base_directory + 'NFL_Defensive.csv'
o_merged_data_path = base_directory + 'NFL_Offensive.csv'
st_merged_data_path = base_directory + 'NFL_Special_Team.csv'
go_merged_data_path = base_directory + 'NFL_Game_Outcome.csv'


# Load each dataset
defensive_data = load_dataset(d_merged_data_path)
offensive_data = load_dataset(o_merged_data_path)
special_teams_data = load_dataset(st_merged_data_path)
game_outcomes_data = load_dataset(go_merged_data_path)

print(game_outcomes_data.head())

# Specify columns to drop for each dataset
defensive_drop_columns = ['Rush', 'YPC', 'RushPct', 'RushPct2', 'FR', 'SFTY', '3rd', '3rdPct', '4th', '4thPct', 'Scrm', 'FF', 'FRPct', 'INTPct', 'INTPct2']
offensive_drop_columns = ['Rush','YPC','RushPct','RushPct2','RushPct3','Rec','Yds','Yds/Rec','RecPct1','RecPct2','RecPct3','Rsh','Tot','2-PT','3rd','3rdPct','4th','4thPct','Scrm']
special_teams_drop_columns = ['1-19 >','20-29 >','30-39 >','40-49 >','50-59 >','60+ >','FGPct','XPM','XP','KRet','PRet','KO','Yds','TB','Ret','RetPct','OSK','OSKPct','OOB','TD','Cmp','CmpPct','Yds/Att','Pass','INT','Rate','1st','1stPct','20+','40+','Sck','SckY','Avg','FC','FUM']
game_outcome_drop_columns = ['Unnamed: 7', 'Day', 'Time']

# Clean each dataset
defensive_data = clean_data(defensive_data, defensive_drop_columns, preserve_col='Team')
offensive_data = clean_data(offensive_data, offensive_drop_columns,  preserve_col='Team')
special_teams_data = clean_data(special_teams_data, special_teams_drop_columns,  preserve_col='Team')
game_outcomes_data = clean_data(game_outcomes_data, game_outcome_drop_columns,  preserve_col='Team')

# Plot correlation matrices for each dataset
plot_correlation_matrix(defensive_data, "Defensive Data Correlation Matrix")
plot_correlation_matrix(offensive_data, "Offensive Data Correlation Matrix")
plot_correlation_matrix(special_teams_data, "Special Teams Data Correlation Matrix")
plot_correlation_matrix(game_outcomes_data, "Game Outcomes Data Correlation Matrix")

def keep_numeric_columns(df):
    return df.select_dtypes(include=[np.number])

# Function to identify and remove highly correlated features
def remove_highly_correlated_features(df, threshold=0.8, exclude_col=None):
    if exclude_col:
        analysis_df = df.drop(columns=[exclude_col])
    else:
        analysis_df = df
    corr_matrix = analysis_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop, axis=1, errors='ignore')
    return df_reduced, to_drop

# Use the functions on the cleaned data
defensive_numeric = keep_numeric_columns(defensive_data)
defensive_data_reduced, defensive_to_drop = remove_highly_correlated_features(defensive_data, exclude_col='Team')

offensive_numeric = keep_numeric_columns(offensive_data)
offensive_data_reduced, offensive_to_drop = remove_highly_correlated_features(offensive_numeric,exclude_col='Team')

special_teams_numeric = keep_numeric_columns(special_teams_data)
special_teams_data_reduced, special_teams_to_drop = remove_highly_correlated_features(special_teams_numeric,exclude_col='Team')

game_outcomes_numeric = keep_numeric_columns(game_outcomes_data)
game_outcomes_data_reduced, game_outcomes_to_drop = remove_highly_correlated_features(game_outcomes_numeric,exclude_col='Team')

# Now you can plot the correlation matrices of the reduced dataframes
plot_correlation_matrix(defensive_data_reduced, "Reduced Defensive Data Correlation Matrix")
plot_correlation_matrix(offensive_data_reduced, "Reduced Offensive Data Correlation Matrix")
plot_correlation_matrix(special_teams_data_reduced, "Reduced Special Teams Data Correlation Matrix")
plot_correlation_matrix(game_outcomes_data_reduced, "Reduced Game Outcomes Data Correlation Matrix")

def apply_pca(df, exclude_col='Team', explained_variance_ratio=0.95):
    if exclude_col in df.columns:
        exclude_data = df[[exclude_col]]
        df = df.drop(columns=[exclude_col])
    else:
        exclude_data = None
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    
    # Apply PCA
    pca = PCA(n_components=explained_variance_ratio)
    principal_components = pca.fit_transform(standardized_data)
    
    # Convert to a DataFrame
    columns = [f'PC{i+1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(data=principal_components, columns=columns)
    
    # Reattach the excluded column if it was separated
    if exclude_data is not None:
        df_pca = pd.concat([exclude_data.reset_index(drop=True), df_pca], axis=1)
    
    return df_pca, pca

# Applying PCA to each dataset while preserving 'Team' column
defensive_data_pca, defensive_pca_model = apply_pca(defensive_data_reduced, 'Team')
print("Defensive Data PCA:")
print(defensive_data_pca.head())

offensive_data_pca, offensive_pca_model = apply_pca(offensive_data_reduced, 'Team')
print("Offensive Data PCA:")
print(offensive_data_pca.head())

special_teams_data_pca, special_teams_pca_model = apply_pca(special_teams_data_reduced, 'Team')
print("Special Teams Data PCA:")
print(special_teams_data_pca.head())

game_outcomes_data_pca, game_outcomes_pca_model = apply_pca(game_outcomes_data_reduced, 'Team')
print("Game Outcomes Data PCA:")
print(game_outcomes_data_pca.head())

# Display the explained variance ratio for each PCA application
print("Defensive Data PCA Explained Variance Ratio:", defensive_pca_model.explained_variance_ratio_)
print("Offensive Data PCA Explained Variance Ratio:", offensive_pca_model.explained_variance_ratio_)
print("Special Teams Data PCA Explained Variance Ratio:", special_teams_pca_model.explained_variance_ratio_)
print("Game Outcomes Data PCA Explained Variance Ratio:", game_outcomes_pca_model.explained_variance_ratio_)

# Merge PCA datasets based on 'Team' column
# This assumes that 'Team' column still exists in each of the PCA datasets
merged_data = game_outcomes_data_pca.merge(defensive_data_pca, on='Team', how='left', suffixes=('', '_def'))
merged_data = merged_data.merge(offensive_data_pca, on='Team', how='left', suffixes=('', '_off'))
merged_data = merged_data.merge(special_teams_data_pca, on='Team', how='left', suffixes=('', '_st'))

# Print the merged dataset
print("Merged Data after PCA:")
print(merged_data.head())
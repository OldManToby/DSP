import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
        df[col] = df[col].fillna(df[col].mean())
    if preserved_data is not None:
        df[preserve_col] = preserved_data
    return df

def assign_seasons_based_on_team_appearances(df, team_col='Team', appearances_per_season=6):

    team_counts = {}
    seasons = []
    
    for team in df[team_col]:
        if team in team_counts:
            team_counts[team] += 1
        else:
            team_counts[team] = 1
        season = 2000 + ((team_counts[team] - 1) // appearances_per_season)
        seasons.append(season)
    df['Season'] = seasons
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


# Specify columns to drop for each dataset
defensive_drop_columns = ['Rush', 'YPC', 'RushPct', 'RushPct2', 'FR', 'SFTY', '3rd', '3rdPct', '4th', '4thPct', 'Scrm', 'FF', 'FRPct', 'INTPct', 'INTPct2']
offensive_drop_columns = ['Rush','YPC','RushPct','RushPct2','RushPct3','Rec','Yds','Yds/Rec','RecPct1','RecPct2','RecPct3','Rsh','Tot','2-PT','3rd','3rdPct','4th','4thPct','Scrm']
special_teams_drop_columns = ['1-19 >','20-29 >','30-39 >','40-49 >','50-59 >','60+ >','FGPct','XPM','XP','KRet','PRet','KO','Yds','TB','Ret','RetPct','OSK','OSKPct','OOB','TD','Cmp','CmpPct','Yds/Att','Pass','INT','Rate','1st','1stPct','20+','40+','Sck','SckY','Avg','FC','FUM']
game_outcome_drop_columns = ['Unnamed: 7', 'Day', 'Time']

game_outcomes_data['Date'] = pd.to_datetime(game_outcomes_data['Date'])
# Adjust the season based on the date, considering NFL season spans from September to January
game_outcomes_data['Season'] = game_outcomes_data['Date'].dt.year
game_outcomes_data.loc[game_outcomes_data['Date'].dt.month < 9, 'Season'] -= 1

# Initialize the records
records = {
    'Team': [],
    'Season': [],
    'Wins': [],
    'Losses': [],
    'PtsW': [],
    'PtsL': [],
    'TOW': [],
    'TOL': []
}

# Process each game
for season, season_df in game_outcomes_data.groupby('Season'):
    for team in pd.concat([season_df['Winner/tie'], season_df['Loser/tie']]).unique():
        # Wins
        wins_df = season_df[season_df['Winner/tie'] == team]
        losses_df = season_df[season_df['Loser/tie'] == team]

        wins = len(wins_df)
        losses = len(losses_df)
        
        pts_for = wins_df['PtsW'].sum() + losses_df['PtsL'].sum()
        pts_against = wins_df['PtsL'].sum() + losses_df['PtsW'].sum()
        
        tow = wins_df['TOW'].sum() + losses_df['TOL'].sum()
        tol = wins_df['TOL'].sum() + losses_df['TOW'].sum()

        records['Team'].append(team)
        records['Season'].append(season)
        records['Wins'].append(wins)
        records['Losses'].append(losses)
        records['PtsW'].append(pts_for)
        records['PtsL'].append(pts_against)
        records['TOW'].append(tow)
        records['TOL'].append(tol)

team_season_stats = pd.DataFrame(records)

print(team_season_stats.head())

# Clean each dataset
defensive_data = clean_data(defensive_data, defensive_drop_columns, preserve_col='Team')
offensive_data = clean_data(offensive_data, offensive_drop_columns,  preserve_col='Team')
special_teams_data = clean_data(special_teams_data, special_teams_drop_columns,  preserve_col='Team')
team_season_stats = clean_data(team_season_stats, columns_to_drop=[],  preserve_col='Team')

defensive_data = assign_seasons_based_on_team_appearances(defensive_data, appearances_per_season=6)
offensive_data = assign_seasons_based_on_team_appearances(offensive_data, appearances_per_season=5)
special_teams_data = assign_seasons_based_on_team_appearances(special_teams_data, appearances_per_season=5)

# Example print to verify the output
print(defensive_data[['Team', 'Season']].tail)
print(offensive_data[['Team', 'Season']].tail)
print(special_teams_data[['Team', 'Season']].tail)
print(team_season_stats[['Team', 'Season']].tail)

# Plot correlation matrices for each dataset
plot_correlation_matrix(defensive_data, "Defensive Data Correlation Matrix")
plot_correlation_matrix(offensive_data, "Offensive Data Correlation Matrix")
plot_correlation_matrix(special_teams_data, "Special Teams Data Correlation Matrix")
plot_correlation_matrix(team_season_stats, "Game Outcomes Data Correlation Matrix")

def keep_numeric_columns(df):
    return df.select_dtypes(include=[np.number])

# Function to identify and remove highly correlated features
def remove_highly_correlated_features(df, threshold=0.8, exclude_cols=[]):
    # Exclude specified columns first
    analysis_df = df.drop(columns=exclude_cols, errors='ignore')
    
    # Ensure that only numeric columns are considered for correlation calculation
    analysis_df = analysis_df.select_dtypes(include=[np.number])
    
    corr_matrix = analysis_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Drop highly correlated columns from the original dataframe
    df_reduced = df.drop(columns=to_drop, errors='ignore')
    return df_reduced, to_drop
exclude_columns = ['Team', 'Season']

# Apply the updated function to your datasets
defensive_data_reduced, defensive_to_drop = remove_highly_correlated_features(defensive_data, exclude_cols=exclude_columns)
offensive_data_reduced, offensive_to_drop = remove_highly_correlated_features(offensive_data, exclude_cols=exclude_columns)
special_teams_data_reduced, special_teams_to_drop = remove_highly_correlated_features(special_teams_data, exclude_cols=exclude_columns)
team_season_stats_data_reduced, team_season_stats_to_drop = remove_highly_correlated_features(team_season_stats, exclude_cols=exclude_columns)


# Now you can plot the correlation matrices of the reduced dataframes
plot_correlation_matrix(defensive_data_reduced, "Reduced Defensive Data Correlation Matrix")
plot_correlation_matrix(offensive_data_reduced, "Reduced Offensive Data Correlation Matrix")
plot_correlation_matrix(special_teams_data_reduced, "Reduced Special Teams Data Correlation Matrix")
plot_correlation_matrix(team_season_stats_data_reduced, "Reduced Game Outcomes Data Correlation Matrix")

def apply_pca(df, exclude_cols=['Team', 'Season'], explained_variance_ratio=0.95):
    exclude_data = df[exclude_cols]
    df = df.drop(columns=exclude_cols, errors='ignore')
    
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df)
    
    pca = PCA(n_components=explained_variance_ratio)
    principal_components = pca.fit_transform(standardized_data)
    
    columns = [f'PC{i+1}' for i in range(pca.n_components_)]
    df_pca = pd.DataFrame(data=principal_components, columns=columns)
    
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

team_season_stats_data_pca, team_season_stats_pca_model = apply_pca(team_season_stats_data_reduced, 'Team')
print("Game Outcomes Data PCA:")
print(team_season_stats_data_pca.head())

# Display the explained variance ratio for each PCA application
print("Defensive Data PCA Explained Variance Ratio:", defensive_pca_model.explained_variance_ratio_)
print("Offensive Data PCA Explained Variance Ratio:", offensive_pca_model.explained_variance_ratio_)
print("Special Teams Data PCA Explained Variance Ratio:", special_teams_pca_model.explained_variance_ratio_)
print("Game Outcomes Data PCA Explained Variance Ratio:", team_season_stats_pca_model.explained_variance_ratio_)

def apply_lda(df, target_col, exclude_cols=['Team', 'Season']):
    # Split the dataframe into features and the target
    X = df.drop(columns=exclude_cols + [target_col], errors='ignore')
    y = df[target_col]
    
    # Standardize the features before applying LDA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create an LDA instance
    lda = LDA(n_components=1)  # n_components should be 1 for binary classification
    # Fit the LDA and transform the data
    X_lda = lda.fit_transform(X_scaled, y)
    
    # Convert the LDA-transformed data back to a DataFrame
    lda_columns = [f'LD{i+1}' for i in range(X_lda.shape[1])]
    df_lda = pd.DataFrame(data=X_lda, columns=lda_columns)
    
    # Add back the excluded and target columns
    df_lda = pd.concat([df[exclude_cols].reset_index(drop=True), df[[target_col]].reset_index(drop=True), df_lda], axis=1)
    
    return df_lda, lda

target_column = 'Att'
defensive_data_lda, defensive_lda_model = apply_lda(defensive_data_reduced, target_column, exclude_cols=['Team', 'Season'])
print("Defensive Data LDA:")
print(defensive_data_lda.head())
offensive_data_lda, offensive_lda_model = apply_lda(offensive_data_reduced, target_column, exclude_cols=['Team', 'Season'])
print("Offensive Data LDA:")
print(offensive_data_lda.head())
special_teams_data_lda, special_teams_lda_model = apply_lda(special_teams_data_reduced, target_column, exclude_cols=['Team', 'Season'])
print("Special Teams Data LDA:")
print(special_teams_data_lda.head())
target_column = 'Wins'
team_season_stats_data_lda, team_season_stats_data_lda_model = apply_lda(team_season_stats_data_reduced, target_column, exclude_cols=['Team', 'Season'])
print("Game Outcome Data LDA:")
print(team_season_stats_data_lda.head())
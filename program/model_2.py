import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , accuracy_score

def train_and_predict(team1, team2):
    def load_defensive(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_defensive(df, drop_columns=None):
        drop_columns = ['Rush', 'YPC', 'RushPct', 'RushPct2', 'FR', 'SFTY', '3rd', '3rdPct',
                        '4th', '4thPct', 'Scrm', 'FF', 'FRPct', 'INTPct', 'INTPct2']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        return df
    def preprocess_defensive(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        df['Team'] = df['Team'].astype('category')
        return df

    def load_offensive(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_offensive(df, drop_columns=None):
        drop_columns = ['Rush','YPC','RushPct','RushPct2','RushPct3','Rec','Yds','Yds/Rec','RecPct1','RecPct2','RecPct3','Rsh','Tot','2-PT','3rd','3rdPct','4th','4thPct','Scrm']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        return df
    def preprocess_offensive(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        df['Team'] = df['Team'].astype('category')
        return df
    def load_st(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_st(df, drop_columns=None):
        drop_columns = ['1-19 >','20-29 >','30-39 >','40-49 >','50-59 >','60+ >','FGPct','XPM','XP','KRet','PRet','KO','Yds','TB','Ret','RetPct','OSK','OSKPct','OOB','TD','Cmp','CmpPct','Yds/Att','Pass','INT','Rate','1st','1stPct','20+','40+','Sck','SckY','Avg','FC','FUM']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        return df
    def preprocess_st(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        df['Team'] = df['Team'].astype('category')
        return df

    def load_game_outcome(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_game_outcome(df, drop_columns=None):
        drop_columns = ['Unnamed: 5', 'Unnamed: 7', 'Day', 'Time']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        df['Date'] = pd.to_datetime(df['Date'])
        df.rename(columns={
            'Winner/tie': 'WinningTeam',
            'Loser/tie': 'LosingTeam',
            'PtsW': 'PointsWon',
            'PtsL': 'PointsLost',
            'YdsW': 'YardsWon',
            'YdsL': 'YardsLost',
            'TOW': 'TurnoversWon',
            'TOL': 'TurnoversLost'
        }, inplace=True)
        
        return df

    def preprocess_game_outcome(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        return df
            

    base_directory = './clean_team_stats/'
    go_merged_data_path = base_directory + 'NFL_Game_Outcome.csv'
    d_merged_data_path = base_directory + 'NFL_Defensive.csv'
    o_merged_data_path = base_directory + 'NFL_Offensive.csv'
    st_merged_data_path = base_directory + 'NFL_Special_Team.csv'

    defensive_data = load_defensive(d_merged_data_path)
    offensive_data = load_offensive(o_merged_data_path)
    st_data = load_st(st_merged_data_path)

    defensive_data = clean_defensive(defensive_data)
    offensive_data = clean_offensive(offensive_data)
    st_data = clean_st(st_data)

    defensive_data = preprocess_defensive(defensive_data)
    offensive_data = preprocess_offensive(offensive_data)
    st_data = preprocess_st(st_data)

    # Assuming 'offensive_data' is your DataFrame for the offensive stats
    offensive_numeric_data = offensive_data.select_dtypes(include=[np.number])
    offensive_correlation_matrix = offensive_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(offensive_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Offensive Data Correlation Matrix")
    plt.show()

    # Assuming 'offensive_data' is your DataFrame for the offensive stats
    defensive_numeric_data = defensive_data.select_dtypes(include=[np.number])
    defensive_correlation_matrix = defensive_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(defensive_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Defensive Data Correlation Matrix")
    plt.show()

    # Assuming 'offensive_data' is your DataFrame for the offensive stats
    st_numeric_data = st_data.select_dtypes(include=[np.number])
    st_correlation_matrix = st_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(st_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("ST Data Correlation Matrix")
    plt.show()

    offensive_avg_stats = offensive_data.groupby('Team', observed=True).mean().reset_index()
    defensive_avg_stats = defensive_data.groupby('Team', observed=True).mean().reset_index()
    st_avg_stats = st_data.groupby('Team', observed=True).mean().reset_index()

    combined_stats = offensive_avg_stats.merge(defensive_avg_stats, on='Team', suffixes=('_off', '_def'))
    combined_stats = combined_stats.merge(st_avg_stats, on='Team')

    go_data = load_game_outcome(go_merged_data_path)
    go_data = clean_game_outcome(go_data)
    go_data = preprocess_game_outcome(go_data)

    def analyze_selected_teams(go_data, combined_stats, team1, team2):
        # Filter games involving either of the selected teams
        relevant_games = go_data[(go_data['WinningTeam'] == team1) | (go_data['WinningTeam'] == team2) | 
                                (go_data['LosingTeam'] == team1) | (go_data['LosingTeam'] == team2)]
        
        differential_features = []
        win_labels = []

        for _, game in relevant_games.iterrows():
            # Determine if team1 is the home team in this game
            is_team1_home = (game['WinningTeam'] == team1 and game['PointsWon'] > game['PointsLost']) or \
                            (game['LosingTeam'] == team1 and game['PointsLost'] > game['PointsWon'])
            
            # Get team stats
            team1_stats = get_team_stats(team1, combined_stats)
            team2_stats = get_team_stats(team2, combined_stats)
            
            if is_team1_home:
                differential = calculate_differentials(team1, team2, combined_stats)
                win_label = game['WinningTeam'] == team1
            else:
                differential = calculate_differentials(team2, team1, combined_stats)
                win_label = game['WinningTeam'] == team2
            
            differential_features.append(differential.flatten())
            win_labels.append(win_label)
        
        # Convert lists to NumPy arrays for model training
        X = np.array(differential_features)
        y = np.array(win_labels).astype(int)  # Convert boolean wins to int for model compatibility
        
        return X, y


    def get_team_stats(team_name, combined_stats):
        return combined_stats[combined_stats['Team'] == team_name]

    def calculate_differentials(home_team, away_team, combined_stats):
        home_stats = combined_stats[combined_stats['Team'] == home_team]
        away_stats = combined_stats[combined_stats['Team'] == away_team]
        differential = home_stats.values - away_stats.values
        return differential

    differential_features = []
    for index, game in go_data.iterrows():
        diff_stats = calculate_differentials(game['Home'], game['Away'], combined_stats)
        differential_features.append(diff_stats)
        
    # Call the function to analyze the selected teams and prepare your feature set (X) and target variable (y) for model training
    X, y = analyze_selected_teams(go_data, combined_stats, team1, team2)

    # With X and y prepared, you can now proceed with model training, validation, and testing as usual:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = rf_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(rf_model, 'rf_model.joblib')

    return rf_model, combined_stats
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
    model = None
    combined_stats = pd.DataFrame()
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
        drop_columns = ['Unnamed: 7', 'Day', 'Time']
        df = df.rename(columns={'Unnamed: 5': 'Home/Away'})
        df['Home/Away'] = df['Home/Away'].fillna('Home')
        df['Home/Away'] = df['Home/Away'].apply(lambda x: 1 if x != '@' else 0)
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

    defensive_stats_team1_team2 = defensive_data[defensive_data['Team'].isin([team1, team2])]
    print("Defensive Stats for Team1 and Team2:")
    print(defensive_stats_team1_team2)
    offensive_stats_team1_team2 = offensive_data[offensive_data['Team'].isin([team1, team2])]
    print("Offensive Stats for Team1 and Team2:")
    print(offensive_stats_team1_team2)
    st_stats_team1_team2 = st_data[st_data['Team'].isin([team1, team2])]
    print("ST Stats for Team1 and Team2:")
    print(st_stats_team1_team2)

    go_data = load_game_outcome(go_merged_data_path)
    go_data = clean_game_outcome(go_data)
    go_data = preprocess_game_outcome(go_data)

    go_stats_team1_team2 = go_stats_team1_team2 = go_data[(go_data['WinningTeam'] == team1) | 
                               (go_data['WinningTeam'] == team2) | 
                               (go_data['LosingTeam'] == team1) | 
                               (go_data['LosingTeam'] == team2)]
    print("go Stats for Team1 and Team2:")
    print(go_stats_team1_team2)


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

    go_numeric_data = go_data.select_dtypes(include=[np.number])
    go_correlation_matrix = go_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(go_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("GO Data Correlation Matrix")
    plt.show()
    return model, combined_stats
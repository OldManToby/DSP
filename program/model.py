import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_predict(team1, team2):
    file_path = 'merged_data.csv'
    data = pd.read_csv(file_path)
    data = data.rename(columns={'Unnamed: 5': 'Home/Away'})
    data['Home/Away'].fillna('Home', inplace=True)
    data['Home/Away'] = data['Home/Away'].apply(lambda x: 1 if x != '@' else 0)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by='Date', inplace=True)

    win_counts = data['Winner/tie'].value_counts()
    loss_counts = data['Loser/tie'].value_counts()

    team_performance = pd.DataFrame({
        'Wins': win_counts,
        'Losses': loss_counts
    }).fillna(0)

    team_performance['WinRate'] = team_performance['Wins'] / (team_performance['Wins'] + team_performance['Losses'])

    data = data.merge(team_performance[['WinRate']], left_on='Winner/tie', right_index=True, how='left')
    data['Team'] = np.where(data['Home/Away'] == 1, data['Winner/tie'], data['Loser/tie'])
    data['Opponent'] = np.where(data['Home/Away'] == 0, data['Winner/tie'], data['Loser/tie'])
    data['Win'] = data['Winner/tie'] == team1

    filtered_data = data[(data['Winner/tie'] == team1) | (data['Loser/tie'] == team1) |
                         (data['Winner/tie'] == team2) | (data['Loser/tie'] == team2)].copy()
    
    filtered_data.drop(columns=['Day','Unnamed: 7'], inplace=True)

    #Calculate wins and losses
    team1_wins = len(filtered_data[filtered_data['Winner/tie'] == team1])
    team1_losses = len(filtered_data[filtered_data['Loser/tie'] == team1])
    team2_wins = len(filtered_data[filtered_data['Winner/tie'] == team2])
    team2_losses = len(filtered_data[filtered_data['Loser/tie'] == team2])
    # Apply the feature engineering functions
    data['RollingAvg_PtsW_' + team1] = calculate_rolling_average(data, team1, 'PtsW', 5)
    data['RollingAvg_PtsW_' + team2] = calculate_rolling_average(data, team2, 'PtsW', 5)
    data['Streak_' + team1] = calculate_streak(data, team1)
    data['Streak_' + team2] = calculate_streak(data, team2)
    data['HeadToHead_WinRate'] = calculate_head_to_head_win_rate(data, team1, team2)

    # Print statements to check if the features are working correctly
    print(f"Rolling average points for {team1}:")
    print(data['RollingAvg_PtsW_' + team1].tail())

    print(f"Rolling average points for {team2}:")
    print(data['RollingAvg_PtsW_' + team2].tail())

    print(f"Win/Loss streak for {team1}:")
    print(data['Streak_' + team1].tail())

    print(f"Win/Loss streak for {team2}:")
    print(data['Streak_' + team2].tail())

    print(f"Head to Head Win Rate between {team1} and {team2}:")
    print(data['HeadToHead_WinRate'].dropna().tail())
    
    #Calculate wins and losses
    team1_wins = len(filtered_data[filtered_data['Winner/tie'] == team1])
    team1_losses = len(filtered_data[filtered_data['Loser/tie'] == team1])
    team2_wins = len(filtered_data[filtered_data['Winner/tie'] == team2])
    team2_losses = len(filtered_data[filtered_data['Loser/tie'] == team2])

    filtered_data.drop(columns=['Date', 'Time', 'Winner/tie', 'Loser/tie'], inplace=True)

    numeric_filtered_data = filtered_data.select_dtypes(include=[np.number])
    
    #Bar plot for individual feature correlation with WinRate
    winrate_correlation = numeric_filtered_data.corrwith(numeric_filtered_data['WinRate']).drop('WinRate')
    plt.figure(figsize=(10, 6))
    winrate_correlation.plot(kind='bar', color='skyblue')
    plt.title(f'Correlation of Features with WinRate for {team1} and {team2}')
    plt.ylabel('Correlation Coefficient')
    plt.show(block=False)

    #Heatmap for all numeric features correlation matrix
    corr_matrix = numeric_filtered_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma', square=True)
    plt.title('Feature Correlation Heatmap for All Numeric Features')
    plt.show(block=False)

    #Boxplot for all numeric features
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=numeric_filtered_data[['WinRate']].dropna())
    plt.title('Win Rate Boxplot')
    plt.ylabel('Win Rate')
    plt.show(block=False)


    print(f"{team1}: {team1_wins} Wins, {team1_losses} Losses")
    print(f"{team2}: {team2_wins} Wins, {team2_losses} Losses")

    summary = f"{team1}: {team1_wins} Wins, {team1_losses} Losses\n"
    summary += f"{team2}: {team2_wins} Wins, {team2_losses} Losses"
    return summary

def calculate_rolling_average(data, team, column, n):
    try:
        team_games = data[(data['Winner/tie'] == team) | (data['Loser/tie'] == team)].copy()
        rolling_avg = team_games[column].rolling(window=n, min_periods=1).mean().iloc[-1]  # Get the last rolling average value
        return rolling_avg
    except Exception as e:
        print(f"Error calculating rolling average for {team}: {e}")
        return np.nan

def calculate_streak(data, team):
    try:
        team_games = data[(data['Winner/tie'] == team) | (data['Loser/tie'] == team)].copy()
        team_games['Win'] = team_games['Winner/tie'] == team
        streaks = team_games['Win'].astype(int).groupby((team_games['Win'] != team_games['Win'].shift()).cumsum()).cumcount() + 1
        current_streak = streaks.iloc[-1] if team_games['Win'].iloc[-1] else -streaks.iloc[-1]  # Positive for win streak, negative for loss streak
        return current_streak
    except Exception as e:
        print(f"Error calculating streak for {team}: {e}")
        return 0

def calculate_head_to_head_win_rate(data, team1, team2):
    try:
        head_to_head_games = data[((data['Winner/tie'] == team1) & (data['Loser/tie'] == team2)) |
                                  ((data['Winner/tie'] == team2) & (data['Loser/tie'] == team1))]
        wins = head_to_head_games[head_to_head_games['Winner/tie'] == team1].shape[0]
        total_games = head_to_head_games.shape[0]
        return wins / total_games if total_games else 0
    except Exception as e:
        print(f"Error calculating head-to-head win rate between {team1} and {team2}: {e}")
        return 0
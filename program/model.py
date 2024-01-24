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

    win_counts = data['Winner/tie'].value_counts()
    loss_counts = data['Loser/tie'].value_counts()

    team_performance = pd.DataFrame({
        'Wins': win_counts,
        'Losses': loss_counts
    }).fillna(0)

    team_performance['WinRate'] = team_performance['Wins'] / (team_performance['Wins'] + team_performance['Losses'])

    data = data.merge(team_performance[['WinRate']], left_on='Winner/tie', right_index=True, how='left')

    filtered_data = data[(data['Winner/tie'] == team1) | (data['Loser/tie'] == team1) |
                         (data['Winner/tie'] == team2) | (data['Loser/tie'] == team2)].copy()
    
    team1_wins = len(filtered_data[filtered_data['Winner/tie'] == team1])
    team1_losses = len(filtered_data[filtered_data['Loser/tie'] == team1])
    team2_wins = len(filtered_data[filtered_data['Winner/tie'] == team2])
    team2_losses = len(filtered_data[filtered_data['Loser/tie'] == team2])

    filtered_data.drop(columns=['Day', 'Date', 'Time', 'Unnamed: 7', 'Winner/tie', 'Loser/tie'], inplace=True)

    numeric_filtered_data = filtered_data.select_dtypes(include=[np.number])
    
    # Create a bar plot for individual feature correlation with WinRate
    winrate_correlation = numeric_filtered_data.corrwith(numeric_filtered_data['WinRate']).drop('WinRate')
    plt.figure(figsize=(10, 6))
    winrate_correlation.plot(kind='bar', color='skyblue')
    plt.title(f'Correlation of Features with WinRate for {team1} and {team2}')
    plt.ylabel('Correlation Coefficient')
    plt.show(block=False)

    # Create a heatmap for all numeric features correlation matrix
    corr_matrix = numeric_filtered_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma', square=True)
    plt.title('Feature Correlation Heatmap for All Numeric Features')
    plt.show(block=False)



    print(f"{team1}: {team1_wins} Wins, {team1_losses} Losses")
    print(f"{team2}: {team2_wins} Wins, {team2_losses} Losses")

    summary = f"{team1}: {team1_wins} Wins, {team1_losses} Losses\n"
    summary += f"{team2}: {team2_wins} Wins, {team2_losses} Losses"
    return summary

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_predict(team1, team2):
    file_path = 'merged_data.csv'
    dfile_path = 'merged_data.csv'
    data = pd.read_csv(file_path)
    data = data.rename(columns={'Unnamed: 5': 'Home/Away'})
    data['Home/Away'].fillna('Home', inplace=True)

    filtered_data = data[(data['Winner/tie'] == team1) | (data['Loser/tie'] == team1) |
                         (data['Winner/tie'] == team2) | (data['Loser/tie'] == team2)]
    
    filtered_data.drop(columns=['Day', 'Date', 'Time', 'Unnamed: 7'], inplace=True)

    print(filtered_data)
    # Tally wins and losses for each team
    team1_wins = len(filtered_data[filtered_data['Winner/tie'] == team1])
    team1_losses = len(filtered_data[filtered_data['Loser/tie'] == team1])
    team2_wins = len(filtered_data[filtered_data['Winner/tie'] == team2])
    team2_losses = len(filtered_data[filtered_data['Loser/tie'] == team2])

    print(f"{team1}: {team1_wins} Wins, {team1_losses} Losses")
    print(f"{team2}: {team2_wins} Wins, {team2_losses} Losses")

    summary = f"{team1}: {team1_wins} Wins, {team1_losses} Losses\n"
    summary += f"{team2}: {team2_wins} Wins, {team2_losses} Losses"
    return summary
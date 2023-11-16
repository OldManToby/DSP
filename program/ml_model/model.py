import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ModelResults:
    def __init__(self, offensive_results_path, defensive_results_path, special_team_results_path):
        # Load data from offensive, defensive, and special teams models
        self.offensive_data = pd.read_csv(offensive_results_path)
        self.defensive_data = pd.read_csv(defensive_results_path)
        self.special_team_data = pd.read_csv(special_team_results_path)

        # Merge the datasets on the 'Team' column
        self.merged_data = pd.merge(self.offensive_data, self.defensive_data, on='Team')
        self.merged_data = pd.merge(self.merged_data, self.special_team_data, on='Team')

        # Assuming 'Outcome' is your target variable
        # Convert the outcome to binary
        self.merged_data['Outcome'] = self.merged_data['Outcome'].map({'Win': 1, 'Loss': 0})

        # Split the data into features (X) and target variable (y)
        self.X = self.merged_data.drop(columns=['Team', 'Outcome'])  # Adjust columns as needed
        self.y = self.merged_data['Outcome']

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        # Train a logistic regression model
        self.logreg_model = LogisticRegression()
        self.logreg_model.fit(self.X_train, self.y_train)

        # Make predictions on the test set
        self.y_pred = self.logreg_model.predict(self.X_test)

        # Evaluate the model
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {self.accuracy}")

def load_team_data(self, away_team, home_team):
     # Load offensive data
    away_team_offensive = self.offensive_data[self.offensive_data['Team'] == away_team].copy()
    home_team_offensive = self.offensive_data[self.offensive_data['Team'] == home_team].copy()
    # Load defensive data
    away_team_defensive = self.defensive_data[self.defensive_data['Team'] == away_team].copy()
    home_team_defensive = self.defensive_data[self.defensive_data['Team'] == home_team].copy()
    # Load special team data
    away_team_special_team = self.special_team_data[self.special_team_data['Team'] == away_team].copy()
    home_team_special_team = self.special_team_data[self.special_team_data['Team'] == home_team].copy()
    # Merge data
    team_data = pd.merge(away_team_offensive, home_team_offensive, on='Team', suffixes=('_away', '_home'))
    team_data = pd.merge(team_data, away_team_defensive, on='Team', suffixes=('_away', '_home'))
    team_data = pd.merge(team_data, home_team_defensive, on='Team', suffixes=('_away', '_home'))
    team_data = pd.merge(team_data, away_team_special_team, on='Team', suffixes=('_away', '_home'))
    team_data = pd.merge(team_data, home_team_special_team, on='Team', suffixes=('_away', '_home'))

    team_data = team_data.drop(columns=['Team_away', 'Team_home'])
    print(f"ModelResults: Loaded data for {away_team} and {home_team}")
    #Return the team_data if needed
    return team_data
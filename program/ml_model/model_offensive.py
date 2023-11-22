# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

class OffensiveModel:
    def __init__(self, folder_path, away_team, home_team):
        self.folder_path = folder_path
        self.away_team = away_team
        self.home_team = home_team
        self.selected_teams_data = None
        self.numerical_features = []
        self.scaler = StandardScaler()
        self.model = LogisticRegression()  # You can change the model as needed

        # Load, preprocess data, and train the model
        self.load_and_preprocess_data()
        self.train_model()

    def load_csv_offensive(self):
        all_data = []
        for subdir, _, files in os.walk(self.folder_path):
            if os.path.basename(subdir).lower() == 'offensive':
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(subdir, file)
                        data = pd.read_csv(file_path)
                        all_data.append(data)
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data

    def load_data(self):
        # Load data from offensive subfolders
        self.selected_teams_data = self.load_csv_offensive()
        if self.selected_teams_data is not None and not self.selected_teams_data.empty:
            print(f"OffensiveModel: Data loaded successfully for {self.away_team} and {self.home_team}")
        else:
            print(f"OffensiveModel: No data available for selected teams: {self.away_team} and {self.home_team}")

    def preprocess_data(self):
        if self.selected_teams_data is not None and not self.selected_teams_data.empty:
            # Handle missing values by filling with the mean of each column
            self.selected_teams_data.fillna(self.selected_teams_data.mean(numeric_only=True), inplace=True)

            # Drop rows containing any remaining missing values
            self.selected_teams_data.dropna(inplace=True)

            # Encoding categorical variables (team names)
            label_encoder = LabelEncoder()
            self.selected_teams_data['Team'] = label_encoder.fit_transform(self.selected_teams_data['Team'])

            # Update numerical features based on columns present in the offensive data
            self.numerical_features = list(self.selected_teams_data.columns[self.selected_teams_data.dtypes == 'float64'])

            # Exclude 'Team' from numerical features if it is in the list
            if 'Team' in self.numerical_features:
                self.numerical_features.remove('Team')

            # Feature Scaling
            self.selected_teams_data[self.numerical_features] = self.scaler.fit_transform(
                self.selected_teams_data[self.numerical_features])

    def display_summary(self):
        # Display summary or any other necessary information
        pass

    def load_and_preprocess_data(self):
        # Load data
        self.load_data()

        # Check if data is loaded successfully
        if self.selected_teams_data is not None and not self.selected_teams_data.empty:
            # Preprocess data
            self.preprocess_data()

            # Display summary
            self.display_summary()

            # Return the processed data
            return self.selected_teams_data

        # If no data is available, return None
        print(f"No data available for selected teams: {self.away_team} and {self.home_team}")
        return None

    def train_model(self):
        # Assuming 'Winning Percentage' is a feature in your data
        self.selected_teams_data['Winning_Percentage_Diff'] = (
            self.selected_teams_data['Home_Team_Winning_Percentage'] - self.selected_teams_data['Away_Team_Winning_Percentage']
        )

        # Assuming 'Win' is the outcome column in your data
        self.selected_teams_data['Outcome'] = (self.selected_teams_data['Winning_Percentage_Diff'] > 0).astype(int)

        # Features (X) and Target Variable (y)
        X = self.selected_teams_data.drop(columns=['Outcome', 'Team'])  # Adjust columns as needed
        y = self.selected_teams_data['Outcome']

        # Feature Selection using SelectKBest
        k_best = 10  # Number of top features to select
        feature_selector = SelectKBest(f_classif, k=k_best)
        X_selected = feature_selector.fit_transform(X, y)

        # Get the selected features
        selected_features = X.columns[feature_selector.get_support()]

        # Train-Test Split on the selected features
        X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)

        # Train a logistic regression model
        logreg_model = LogisticRegression()
        logreg_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = logreg_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")


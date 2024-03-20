import sys
import os
import random
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , classification_report
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QComboBox, QPushButton, QLabel, QStyleFactory, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from team_names import team_names

class DisclaimerDialog(QMessageBox):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Disclaimer')
        self.setText("This tool is for research purposes only and should not be used for any form of gambling.\n\nThe data used a presented is provided by The Football Database(www.footballdb.com) & contains statistics from 2000 onwards.\n"
                      "\nThe accuracy of this model is not guaranteed.\n\n"
                      "By clicking 'Acknowledge,' you agree to use this tool responsibly.")
        self.setIcon(QMessageBox.Information)
        self.addButton(QPushButton('Acknowledge'), QMessageBox.AcceptRole)

class PredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.team_logos = {}
        self.model, self.scaler, self.lda, self.merged_data = self.train_and_predict()
        self.initUI()


    def initUI(self):
        disclaimer_dialog = DisclaimerDialog()
        disclaimer_dialog.exec_()
        self.setWindowTitle('NFL Match Predictor')
        self.setGeometry(100,100,600,450)
        app_icon = QIcon('NFL_Logo.jpg')
        self.setWindowIcon(app_icon)
        layout = QGridLayout()

        self.away_team_logo = QLabel(self)
        self.home_team_logo = QLabel(self)

        away_label = QLabel('Away Team:')
        home_label = QLabel('Home Team:')

        # Construct team logos dictionary using team_names keys
        team_logos_dir = 'nfl_teams'
        for folder_name, team_list in team_names.items():
        # Use only the single-word team name for logo mapping
            single_word_team_name = team_list[-1]  # Gets the last name, which is used in your datasets
            logo_path = os.path.join(team_logos_dir, folder_name, f'{folder_name.lower()}.png')
            self.team_logos[single_word_team_name] = logo_path

        # Populate available teams
        self.available_teams = list(self.team_logos.keys())

        # Dropdown menu for home/away team
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItems(self.available_teams)

        self.away_team_combo = QComboBox()
        self.away_team_combo.addItems(self.available_teams)

        self.home_team_combo.currentIndexChanged.connect(self.updateLogos)
        self.away_team_combo.currentIndexChanged.connect(self.updateLogos)

        self.home_team_combo.setCurrentText(random.choice(self.available_teams))
        self.away_team_combo.setCurrentText(random.choice(self.available_teams))

        self.updateLogos()

        predict_button = QPushButton('Predict')
        predict_button.clicked.connect(self.on_predict_button_clicked)

        self.result_label = QLabel('Prediction will be shown here.')
        self.result_label.setWordWrap(True)
        self.result_label.setFixedSize(300, 60)

        layout.addWidget(self.away_team_logo,0,0,2,2)
        layout.addWidget(self.home_team_logo,0,2,2,2)

        layout.addWidget(away_label,2,0)
        layout.addWidget(self.away_team_combo,2,1)
        layout.addWidget(home_label,2,2)
        layout.addWidget(self.home_team_combo,2,3)
        layout.addWidget(predict_button,3,1,1,2)
        layout.addWidget(self.result_label,4,0,2,2)

        self.setLayout(layout)

    def updateLogos(self):
        away_team = self.away_team_combo.currentText()
        home_team = self.home_team_combo.currentText()

        away_logo_path = self.team_logos.get(away_team, '')
        home_logo_path = self.team_logos.get(home_team, '')

        if away_logo_path:
            away_pixmap = QPixmap(away_logo_path).scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            if not away_pixmap.isNull():
                self.away_team_logo.setPixmap(away_pixmap)
                self.away_team_logo.setStyleSheet("border: 2px solid black;")  # Set border style
            else:
                print("Error loading away team logo image")
        else:
            self.away_team_logo.clear()
        if home_logo_path:
            home_pixmap = QPixmap(home_logo_path).scaled(300, 300, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            if not home_pixmap.isNull():
                self.home_team_logo.setPixmap(home_pixmap)
                self.home_team_logo.setStyleSheet("border: 2px solid black;")
            else:
                print("Error loading home team logo image")
        else:
            self.home_team_logo.clear()

        # Disable selected teams in the other dropdown menu
        selected_home_team = self.home_team_combo.currentText()
        selected_away_team = self.away_team_combo.currentText()

        self.away_team_combo.blockSignals(True)
        self.away_team_combo.clear()
        self.away_team_combo.addItems([team for team in self.available_teams if team != selected_home_team])
        self.away_team_combo.setCurrentText(selected_away_team)
        self.away_team_combo.blockSignals(False)

        self.home_team_combo.blockSignals(True)
        self.home_team_combo.clear()
        self.home_team_combo.addItems([team for team in self.available_teams if team != selected_away_team])
        self.home_team_combo.setCurrentText(selected_home_team)
        self.home_team_combo.blockSignals(False)

    
    def predict_winner(self, team1, team2):
        try:
            # Extracting teams data
            teams_data = self.merged_data[(self.merged_data['Team'] == team1) | (self.merged_data['Team'] == team2)]
            
            if len(teams_data) != 2:
                return "Error: One or both teams not found in the dataset."
            
            # Drop columns not present during the scaler fitting
            features = teams_data.drop(['Team', 'Season', 'Wins', 'WinningSeason', 'LDA_Component'], axis=1, errors='ignore')
            
            # Check if there are sufficient features for prediction
            if len(features.columns) < 1:
                return "Error: Insufficient features for prediction."
            
            # Proceed with the scaling and LDA transformation as before
            features_scaled = self.scaler.transform(features)
            features_lda = self.lda.transform(features_scaled)
            
            # Prediction logic as before
            prediction = self.model.predict(features_lda)
            
            if prediction[0] == 1:
                return team1 + " are predicted to win"
            else:
                return team2 + " are predicted to win"
        
        except Exception as e:
            return f"Error: {str(e)}"

    def on_predict_button_clicked(self):
        team1 = self.home_team_combo.currentText()
        team2 = self.away_team_combo.currentText()
        
        # Predict the winner
        winner_prediction = self.predict_winner(team1, team2)
        
        # Generate detailed comparison
        comparison_report = self.generate_comparison_report(team1, team2)
        
        # Update the result label with the winner prediction
        self.result_label.setText(winner_prediction)

    def generate_comparison_report(self, team1, team2):
        # Get data for the selected teams
        team1_data = self.merged_data[self.merged_data['Team'] == team1].iloc[0]
        team2_data = self.merged_data[self.merged_data['Team'] == team2].iloc[0]
        
        # Define features to compare (retrieve all numeric columns except 'Team')
        features = [col for col in self.merged_data.columns if col != 'Team' and self.merged_data[col].dtype != 'object']
        
        # Create a single graph for all feature comparisons
        plt.figure(figsize=(10, 6))
        
        # Generate scatter plots for each feature
        for feature in features:
            plt.scatter([team1, team2], [team1_data[feature], team2_data[feature]], label=feature)
        
        # Add labels, title, legend, and grid
        plt.xlabel('Team')
        plt.ylabel('Value')
        plt.title(f'Comparison of Features between {team1} and {team2}')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Save the comparison plot
        plt.tight_layout()
        plt.savefig('comparison_features.png')
        
        # Generate a PDF report with the comparison plot and other information
        comparison_report = f"Comparison Report:\n\n{team1} vs {team2}\n\n"
        comparison_report += "Feature Comparisons:\n"
        for feature in features:
            comparison_report += f"{feature}: {team1_data[feature]} vs {team2_data[feature]}\n"
        
        # Add more analysis and explanations
        comparison_report += "\nDecision Analysis:\n"
        for feature in features:
            if team1_data[feature] > team2_data[feature]:
                comparison_report += f"{team1} has a higher {feature} compared to {team2}.\n"
            elif team1_data[feature] < team2_data[feature]:
                comparison_report += f"{team2} has a higher {feature} compared to {team1}.\n"
            else:
                comparison_report += f"{team1} and {team2} have the same {feature}.\n"
        
        return comparison_report


    def train_and_predict(self):
        def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model'):
                try:
                    # For logistic regression with GridSearchCV
                    if model_name == "Logistic Regression (GridSearchCV)":
                        model.fit(X_train, y_train)
                        best_model = model.best_estimator_
                        y_pred = best_model.predict(X_test)
                        print(f"Results for {model_name}:")
                        print("Best Parameters:", model.best_params_)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        print(f"Results for {model_name}:")

                    print(classification_report(y_test, y_pred))
                    print("Accuracy:", accuracy_score(y_test, y_pred))
                    print("-----------\n")
                except Exception as e:
                    print(f"An error occurred while training the {model_name}: {e}")

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

        # Assuming 'Team' and 'Season' are in all datasets and are of the same format
        merged_data = defensive_data.merge(offensive_data, on=['Team', 'Season'], suffixes=('_def', '_off'))
        merged_data = merged_data.merge(special_teams_data, on=['Team', 'Season'], suffixes=('', '_st'))
        merged_data = merged_data.merge(team_season_stats[['Team', 'Season', 'Wins']], on=['Team', 'Season'])

        # Example: Define a winning season based on the median number of wins
        median_wins = merged_data['Wins'].median()
        merged_data['WinningSeason'] = (merged_data['Wins'] > median_wins).astype(int)

        # Exclude 'Team' and 'Season' for LDA, and use 'WinningSeason' as the target
        features = merged_data.drop(['Team', 'Season', 'Wins', 'WinningSeason'], axis=1)
        target = merged_data['WinningSeason']

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply LDA
        lda = LDA(n_components=1)  # Using 1 as this is a binary classification
        features_lda = lda.fit_transform(features_scaled, target)

        # You can add the LDA component back to the dataframe if you want to visualize or further analyze
        merged_data['LDA_Component'] = features_lda

        X_train, X_test, y_train, y_test = train_test_split(features_lda, target, test_size=0.3, random_state=42)

        # Training a simple model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predicting and evaluating the model
        predictions = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, predictions))

        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }
        grid_search_lr = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

        # Initialize models outside of the function
        models = {
            "Logistic Regression (GridSearchCV)": grid_search_lr,
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Neural Network": MLPClassifier(random_state=42, max_iter=1000),
        }

        # Call train_and_evaluate_model for each model
        for model_name, model in models.items():
            train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)

        return model , scaler , lda , merged_data


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
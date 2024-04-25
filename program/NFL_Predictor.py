import sys
import os
import random
import numpy as np
import pandas as pd
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
        exclude_columns = ['Team', 'Season', 'Wins', 'WinningSeason', 'LDA_Component'] 
        teams_data = self.merged_data[(self.merged_data['Team'] == team1) | (self.merged_data['Team'] == team2)]
        features = teams_data.drop(exclude_columns, axis=1)
        features_scaled = self.scaler.transform(features)
        features_lda = self.lda.transform(features_scaled)

        # Get the prediction and probabilities once
        probabilities = self.model.predict_proba(features_lda)
        prediction = self.model.predict(features_lda)

        # Calculate probabilities
        team1_probability = probabilities[0][1] * 100
        team2_probability = probabilities[0][0] * 100

        def plot_best_features(teams_data, team1, team2):
            # Determine the best feature based on the highest value for simplicity; adjust logic as needed
            best_feature_team1 = teams_data.loc[teams_data['Team'] == team1].drop(columns=['Team', 'Season', 'Wins', 'WinningSeason', 'LDA_Component']).idxmax(axis=1).values[0]
            best_value_team1 = teams_data.loc[teams_data['Team'] == team1, best_feature_team1].values[0]
            
            best_feature_team2 = teams_data.loc[teams_data['Team'] == team2].drop(columns=['Team', 'Season', 'Wins', 'WinningSeason', 'LDA_Component']).idxmax(axis=1).values[0]
            best_value_team2 = teams_data.loc[teams_data['Team'] == team2, best_feature_team2].values[0]
            
            # Create a DataFrame for the plot
            data = {
                'Statistic': [best_feature_team1, best_feature_team2],
                'Value': [best_value_team1, best_value_team2],
                'Team': [team1, team2]
            }
            df = pd.DataFrame(data)
            
            # Create a bar plot
            plt.figure(figsize=(10, 6))
            barplot = sns.barplot(x='Statistic', y='Value', hue='Team', data=df)
            plt.title('Best Feature Comparison Between Teams')
            plt.ylabel('Value')
            plt.xlabel('Best Feature')
            plt.tight_layout()
            plt.show()

        # Example usage
        plot_best_features(teams_data, team1, team2)

        if prediction[0] == 1:
            return f"{team1} are predicted to win with a possibility of {team1_probability:.2f}%"
        else:
            return f"{team2} are predicted to win with a possibility of {team2_probability:.2f}%"

    def on_predict_button_clicked(self):
        team1 = self.home_team_combo.currentText()
        team2 = self.away_team_combo.currentText()
        # Predict the winner
        winner_prediction = self.predict_winner(team1, team2)
        self.result_label.setText(winner_prediction)

# Model For NFL Predictor
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

        
        def clean_data(df, columns_to_drop, preserve_col=None):
            preserved_data = None
            if preserve_col and preserve_col in df.columns:
                preserved_data = df[preserve_col].reset_index(drop=True)
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            
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

        def plot_correlation_matrix(df, title):
            
            numeric_df = df.select_dtypes(include=[np.number])
            plt.figure(figsize=(12, 10))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='magma')
            plt.title(title)
            plt.show()

        def extract_home_away_features(df):
            # Initialize columns for home and away team stats
            df['HomeTeam'] = None
            df['AwayTeam'] = None
            df['HomePts'] = 0
            df['AwayPts'] = 0
            df['HomeYds'] = 0
            df['AwayYds'] = 0
            df['HomeTO'] = 0
            df['AwayTO'] = 0
            
            # Assign values based on whether the winner was at home or away
            for idx, row in df.iterrows():
                if row['Unnamed: 5'] == '@':  # Winner is the away team
                    df.at[idx, 'AwayTeam'] = row['Winner/tie']
                    df.at[idx, 'HomeTeam'] = row['Loser/tie']
                    df.at[idx, 'AwayPts'] = row['PtsW']
                    df.at[idx, 'HomePts'] = row['PtsL']
                    df.at[idx, 'AwayYds'] = row['YdsW']
                    df.at[idx, 'HomeYds'] = row['YdsL']
                    df.at[idx, 'AwayTO'] = row['TOW']
                    df.at[idx, 'HomeTO'] = row['TOL']
                else:  # Winner is the home team
                    df.at[idx, 'HomeTeam'] = row['Winner/tie']
                    df.at[idx, 'AwayTeam'] = row['Loser/tie']
                    df.at[idx, 'HomePts'] = row['PtsW']
                    df.at[idx, 'AwayPts'] = row['PtsL']
                    df.at[idx, 'HomeYds'] = row['YdsW']
                    df.at[idx, 'AwayYds'] = row['YdsL']
                    df.at[idx, 'HomeTO'] = row['TOW']
                    df.at[idx, 'AwayTO'] = row['TOL']

            return df

        base_directory = './clean_team_stats/'
        d_merged_data_path = base_directory + 'NFL_Defensive.csv'
        o_merged_data_path = base_directory + 'NFL_Offensive.csv'
        st_merged_data_path = base_directory + 'NFL_Special_Team.csv'
        go_merged_data_path = base_directory + 'NFL_Game_Outcome.csv'

        defensive_data = load_dataset(d_merged_data_path)
        offensive_data = load_dataset(o_merged_data_path)
        special_teams_data = load_dataset(st_merged_data_path)
        game_outcomes_data = load_dataset(go_merged_data_path)
        game_outcomes_data = extract_home_away_features(game_outcomes_data)

        defensive_drop_columns = ['Rush', 'YPC', 'RushPct', 'RushPct2', 'FR', 'SFTY', '3rd', '3rdPct', '4th', '4thPct', 'Scrm', 'FF', 'FRPct', 'INTPct', 'INTPct2']
        offensive_drop_columns = ['Rush','YPC','RushPct','RushPct2','RushPct3','Rec','Yds','Yds/Rec','RecPct1','RecPct2','RecPct3','Rsh','Tot','2-PT','3rd','3rdPct','4th','4thPct','Scrm']
        special_teams_drop_columns = ['1-19 >','20-29 >','30-39 >','40-49 >','50-59 >','60+ >','FGPct','XPM','XP','KRet','PRet','KO','Yds','TB','Ret','RetPct','OSK','OSKPct','OOB','TD','Cmp','CmpPct','Yds/Att','Pass','INT','Rate','1st','1stPct','20+','40+','Sck','SckY','Avg','FC','FUM']
        game_outcome_drop_columns = ['Unnamed: 7', 'Day', 'Time']

        game_outcomes_data['Date'] = pd.to_datetime(game_outcomes_data['Date'])
        game_outcomes_data['Season'] = game_outcomes_data['Date'].dt.year
        game_outcomes_data.loc[game_outcomes_data['Date'].dt.month < 9, 'Season'] -= 1
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

        for season, season_df in game_outcomes_data.groupby('Season'):
            for team in pd.concat([season_df['Winner/tie'], season_df['Loser/tie']]).unique():

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


        defensive_data = clean_data(defensive_data, defensive_drop_columns, preserve_col='Team')
        offensive_data = clean_data(offensive_data, offensive_drop_columns,  preserve_col='Team')
        special_teams_data = clean_data(special_teams_data, special_teams_drop_columns,  preserve_col='Team')
        team_season_stats = clean_data(team_season_stats, columns_to_drop=[],  preserve_col='Team')

        defensive_data = assign_seasons_based_on_team_appearances(defensive_data, appearances_per_season=6)
        offensive_data = assign_seasons_based_on_team_appearances(offensive_data, appearances_per_season=5)
        special_teams_data = assign_seasons_based_on_team_appearances(special_teams_data, appearances_per_season=5)

        plot_correlation_matrix(defensive_data, "Defensive Data Correlation Matrix")
        plot_correlation_matrix(offensive_data, "Offensive Data Correlation Matrix")
        plot_correlation_matrix(special_teams_data, "Special Teams Data Correlation Matrix")
        plot_correlation_matrix(team_season_stats, "Game Outcomes Data Correlation Matrix")

        def keep_numeric_columns(df):
            return df.select_dtypes(include=[np.number])

        def remove_highly_correlated_features(df, threshold=0.8, exclude_cols=[]):

            analysis_df = df.drop(columns=exclude_cols, errors='ignore')
            analysis_df = analysis_df.select_dtypes(include=[np.number])
            
            corr_matrix = analysis_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            df_reduced = df.drop(columns=to_drop, errors='ignore')
            return df_reduced, to_drop
        exclude_columns = ['Team', 'Season', 'Wins']

        defensive_data_reduced, defensive_to_drop = remove_highly_correlated_features(defensive_data, exclude_cols=exclude_columns)
        offensive_data_reduced, offensive_to_drop = remove_highly_correlated_features(offensive_data, exclude_cols=exclude_columns)
        special_teams_data_reduced, special_teams_to_drop = remove_highly_correlated_features(special_teams_data, exclude_cols=exclude_columns)
        team_season_stats_data_reduced, team_season_stats_to_drop = remove_highly_correlated_features(team_season_stats, exclude_cols=exclude_columns)

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

        print("Defensive Data PCA Explained Variance Ratio:", defensive_pca_model.explained_variance_ratio_)
        print("Offensive Data PCA Explained Variance Ratio:", offensive_pca_model.explained_variance_ratio_)
        print("Special Teams Data PCA Explained Variance Ratio:", special_teams_pca_model.explained_variance_ratio_)
        print("Game Outcomes Data PCA Explained Variance Ratio:", team_season_stats_pca_model.explained_variance_ratio_)

        def apply_lda(df, target_col, exclude_cols=['Team', 'Season']):

            X = df.drop(columns=exclude_cols + [target_col], errors='ignore')
            y = df[target_col]           

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)           
            lda = LDA(n_components=1)

            X_lda = lda.fit_transform(X_scaled, y)
            lda_columns = [f'LD{i+1}' for i in range(X_lda.shape[1])]
            df_lda = pd.DataFrame(data=X_lda, columns=lda_columns)
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

        merged_data = defensive_data.merge(offensive_data, on=['Team', 'Season'], suffixes=('_def', '_off'))
        merged_data = merged_data.merge(special_teams_data, on=['Team', 'Season'], suffixes=('', '_st'))
        merged_data = merged_data.merge(team_season_stats[['Team', 'Season', 'Wins']], on=['Team', 'Season'])
        median_wins = merged_data['Wins'].median()
        merged_data['WinningSeason'] = (merged_data['Wins'] > median_wins).astype(int)
        features = merged_data.drop(['Team', 'Season', 'Wins', 'WinningSeason'], axis=1)
        target = merged_data['WinningSeason']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        lda = LDA(n_components=1)
        features_lda = lda.fit_transform(features_scaled, target)
        merged_data['LDA_Component'] = features_lda

        X_train, X_test, y_train, y_test = train_test_split(features_lda, target, test_size=0.3, random_state=42)
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

        models = {
            "RandomForestClassifier": RandomForestClassifier(),
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
    app.setStyleSheet("""
                QWidget {
            background-color: #013369;
            color: white;
            font-family: "Segoe UI", Arial, sans-serif;
        }

        QPushButton {
            background-color: #d50a0a;
            color: white;
            border: 1px solid #d50a0a;
            padding: 10px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
        }

        QPushButton:hover {
            background-color: #f53b3b;
            border-color: #f53b3b;
        }

        QPushButton:pressed {
            background-color: #b30909;
        }

        QLabel {
            color: white;
            font-size: 14px;
            padding: 2px;
        }

        QComboBox {
            background-color: white;
            color: #013369;
            border: 1px solid white;
            border-radius: 3px;
            padding: 5px;
            font-size: 14px;
            min-width: 150px;
        }

        QComboBox QAbstractItemView {
            background-color: white;
            selection-background-color: #d50a0a;
            color: #013369;
            border-radius: 3px;
        }

        QMessageBox {
            background-color: #013369;
        }

        QMessageBox QPushButton {
            background-color: #d50a0a;
            border-radius: 3px;
            padding: 5px;
            margin: 5px;
            color: white;
        }

        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 30px;
            border-left-width: 1px;
            border-left-color: white;
            border-left-style: solid;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;
        }
    """)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())
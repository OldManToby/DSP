import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from joblib import dump

def preprocess_data(data, away_team=None, home_team=None):
    data['Away_Team'] = (data['Loser/tie'] == away_team) | (data['Winner/tie'] == away_team)
    data['Home_Team'] = (data['Winner/tie'] == home_team) | (data['Loser/tie'] == home_team)

    data = data.rename(columns={'Unnamed: 5': 'Home/Away'})
    data['Date'] = pd.to_datetime(data['Date'])
    data.drop(columns=['Day', 'Date', 'Time', 'Unnamed: 7', 'Winner/tie', 'Loser/tie'], inplace=True)
    data['Home_Win'] = data['Home/Away'].apply(lambda x: 0 if x == '@' else 1)
    data.drop('Home/Away', axis=1, inplace=True)

    data['Point_Differential'] = data['PtsW'] - data['PtsL']
    data['Yard_Differential'] = data['YdsW'] - data['YdsL']
    data['Turnover_Differential'] = data['TOW'] - data['TOL']

    minmax_scaler = MinMaxScaler()
    data = pd.DataFrame(minmax_scaler.fit_transform(data), columns=data.columns)
    return data


folder_path = 'team_stats/Game_Outcome'
predictions_dict = {}

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
best_scores = []
classifiers = []
param_distributions = {
    'n_estimators': [1, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'max_depth': [None, 1, 5, 10, 20, 25, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7],
    'min_impurity_decrease': [0.0, 0.01, 0.1],
    'bootstrap': [True, False],
    'class_weight':[None, 'balanced']
    }

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_distributions, n_iter=200, cv=2, verbose=0, random_state=42, n_jobs=-1)

for i, csv_file in enumerate(csv_files):
    file_path = os.path.join(folder_path, csv_file)
    data = pd.read_csv(file_path)
    data = data.rename(columns={'Unnamed: 5': 'Home/Away'})
    data = preprocess_data(data)
    
    X = data.drop('Home_Win', axis=1)
    y = data['Home_Win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_search.fit(X_train, y_train)
    best_score = round(random_search.best_score_, 2)
    best_scores.append(best_score)
    best_params = random_search.best_params_
    best_rf_model = RandomForestClassifier(**best_params, random_state=42)
    
    best_rf_model.fit(X_train, y_train)
    best_params_rounded = {}
    for param, value in random_search.best_params_.items():
        if isinstance(value, (int, float)):
            best_params_rounded[param] = round(value, 2)
        else:
            best_params_rounded[param] = value
    print(f"Best Parameters for {csv_file}: {best_params_rounded}")
    print(f"Best Score for {csv_file}: {round(random_search.best_score_, 2)}")

    best_rf_model = random_search.best_estimator_
    
    classifier_name = f"RF{i + 1}"
    classifiers.append((classifier_name, best_rf_model))

if best_scores:
    overall_score = sum(best_scores) / len(best_scores)
    print("Individual Scores:")
    for i, csv_file in enumerate(csv_files):
        print(f"{csv_file}: {best_scores[i]}")
    print(f"Mean Average Score: {overall_score:.2f}")
else:
    print("No scores to calculate. Check your data or file paths.")

dump(best_rf_model,'results.txt')
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the data
file_path = 'merged_data.csv'
data = pd.read_csv(file_path)
data = data.rename(columns={'Unnamed: 5': 'Home/Away'})
data['Date'] = pd.to_datetime(data['Date'])

# Basic preprocessing
data.drop(columns=['Day','Date','Time', 'Unnamed: 7'], inplace=True)
data['Home_Win'] = data['Home/Away'].apply(lambda x: 0 if x == '@' else 1)
# Drop the 'Home/Away' column from the dataset
data.drop('Home/Away', axis=1, inplace=True)
data['Point_Differential'] = data['PtsW'] - data['PtsL']
data['Yard_Differential'] = data['YdsW'] - data['YdsL']
data['Turnover_Differential'] = data['TOW'] - data['TOL']

# One-hot encode the 'Winner/tie' and 'Loser/tie' columns
winner_dummies = pd.get_dummies(data['Winner/tie'], prefix='Winner')
loser_dummies = pd.get_dummies(data['Loser/tie'], prefix='Loser')

# Concatenate the dummy variables with your DataFrame and drop the original columns
data = pd.concat([data, winner_dummies, loser_dummies], axis=1)
data.drop(['Winner/tie', 'Loser/tie'], axis=1, inplace=True)

# Selecting features and target variable for the model
X = data.drop('Home_Win', axis=1)
y = data['Home_Win']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=2, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X, y)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the results
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)

# Train the model with the best parameters on the training set
best_rf_model = RandomForestClassifier(random_state=42, **best_params)
best_rf_model.fit(X_train, y_train)

# Predicting and evaluating the model
predictions = best_rf_model.predict(X_test)
print(classification_report(y_test, predictions))

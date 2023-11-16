import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data from offensive, defensive, and special teams models
offensive_data = pd.read_csv('path_to_offensive_model_results.csv')
defensive_data = pd.read_csv('path_to_defensive_model_results.csv')
special_team_data = pd.read_csv('path_to_special_team_model_results.csv')

# Assuming 'Team' is a common column in all three datasets
# Merge the datasets on the 'Team' column
merged_data = pd.merge(offensive_data, defensive_data, on='Team')
merged_data = pd.merge(merged_data, special_team_data, on='Team')

# Assuming 'Outcome' is your target variable
# Convert the outcome to binary
merged_data['Outcome'] = merged_data['Outcome'].map({'Win': 1, 'Loss': 0})

# Split the data into features (X) and target variable (y)
X = merged_data.drop(columns=['Team', 'Outcome'])  # Adjust columns as needed
y = merged_data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Now you can use this trained model to make predictions on new data
# new_data = ...  # Load your new data
# new_predictions = logreg_model.predict(new_data)

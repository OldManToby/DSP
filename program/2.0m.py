import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Define a function to load and clean a CSV file
def load_and_clean_csv(file_path):
    # Load the CSV file, specifying ',' as an additional NaN value
    df = pd.read_csv(file_path, na_values=[','])

    if 'SpecialColumn' in df.columns:
        df[['Metric1', 'Metric2']] = df['SpecialColumn'].str.split('_', expand=True).astype(float)
    
    # Convert date columns to datetime objects for time series analysis
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    df.rename(columns={'Unnamed: 5': 'Home/Away'}, inplace=True)
    columns_to_drop = ['Day', 'Time', 'Unnamed: 7', 'Lng', 'TBPct']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    return df

base_directory = './team_stats/'
go_merged_data_path = base_directory + 'NFL_Game_Outcome.csv'
d_merged_data_path = base_directory + 'NFL_Defensive.csv'
o_merged_data_path = base_directory + 'NFL_Offensive.csv'
st_merged_data_path = base_directory + 'NFL_Special_Team.csv'

go_merged_data = load_and_clean_csv(go_merged_data_path)
d_merged_data = load_and_clean_csv(d_merged_data_path)
o_merged_data = load_and_clean_csv(o_merged_data_path)
st_merged_data = load_and_clean_csv(st_merged_data_path)

new_directory = './clean_team_stats/'
if not os.path.exists(new_directory):
    os.makedirs(new_directory)
go_merged_data.to_csv(new_directory + 'NFL_Game_Outcome.csv', index=False)
d_merged_data.to_csv(new_directory + 'NFL_Defensive.csv', index=False)
o_merged_data.to_csv(new_directory + 'NFL_Offensive.csv', index=False)
st_merged_data.to_csv(new_directory + 'NFL_Special_Team.csv', index=False)

# Now you can proceed with feature engineering, model training, etc.

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix

def train_and_predict(team1, team2):
    model = None
    combined_stats = pd.DataFrame()

    # Begin Loading, Cleaning and Preproccessing of all datasets
    def load_defensive(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_defensive(df, drop_columns=None):
        drop_columns = ['Rush', 'YPC', 'RushPct', 'RushPct2', 'FR', 'SFTY', '3rd', '3rdPct',
                        '4th', '4thPct', 'Scrm', 'FF', 'FRPct', 'INTPct', 'INTPct2']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        return df
    def preprocess_defensive(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        df['Team'] = df['Team'].astype('category')
        return df

    def load_offensive(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_offensive(df, drop_columns=None):
        drop_columns = ['Rush','YPC','RushPct','RushPct2','RushPct3','Rec','Yds','Yds/Rec','RecPct1','RecPct2','RecPct3','Rsh','Tot','2-PT','3rd','3rdPct','4th','4thPct','Scrm']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        return df
    def preprocess_offensive(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        df['Team'] = df['Team'].astype('category')
        return df
    def load_st(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_st(df, drop_columns=None):
        drop_columns = ['1-19 >','20-29 >','30-39 >','40-49 >','50-59 >','60+ >','FGPct','XPM','XP','KRet','PRet','KO','Yds','TB','Ret','RetPct','OSK','OSKPct','OOB','TD','Cmp','CmpPct','Yds/Att','Pass','INT','Rate','1st','1stPct','20+','40+','Sck','SckY','Avg','FC','FUM']
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        return df
    def preprocess_st(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        df['Team'] = df['Team'].astype('category')
        return df

    def load_game_outcome(file_path, na_values=None):
        return pd.read_csv(file_path, na_values=na_values)
    def clean_game_outcome(df, drop_columns=None):
        drop_columns = ['Unnamed: 7', 'Day', 'Time']
        df = df.rename(columns={'Unnamed: 5': 'Home/Away'})
        df['Home/Away'] = df['Home/Away'].fillna('Home')
        df['Home/Away'] = df['Home/Away'].apply(lambda x: 1 if x != '@' else 0)
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        df['Date'] = pd.to_datetime(df['Date'])
        df.rename(columns={
            'Winner/tie': 'WinningTeam',
            'Loser/tie': 'LosingTeam',
            'PtsW': 'PointsWon',
            'PtsL': 'PointsLost',
            'YdsW': 'YardsWon',
            'YdsL': 'YardsLost',
            'TOW': 'TurnoversWon',
            'TOL': 'TurnoversLost'
        }, inplace=True)
        
        return df

    def preprocess_game_outcome(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float32']).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        return df
    
    base_directory = './clean_team_stats/'
    go_merged_data_path = base_directory + 'NFL_Game_Outcome.csv'
    d_merged_data_path = base_directory + 'NFL_Defensive.csv'
    o_merged_data_path = base_directory + 'NFL_Offensive.csv'
    st_merged_data_path = base_directory + 'NFL_Special_Team.csv'

    defensive_data = load_defensive(d_merged_data_path)
    offensive_data = load_offensive(o_merged_data_path)
    st_data = load_st(st_merged_data_path)

    defensive_data = clean_defensive(defensive_data)
    offensive_data = clean_offensive(offensive_data)
    st_data = clean_st(st_data)

    defensive_data = preprocess_defensive(defensive_data)
    offensive_data = preprocess_offensive(offensive_data)
    st_data = preprocess_st(st_data)

    #Ensuring teams selected are filtered through the data therefore only displaying the teams the user has chosen
    defensive_stats_team1_team2 = defensive_data[defensive_data['Team'].isin([team1, team2])]
    print("Defensive Stats for Team1 and Team2:")
    print(defensive_stats_team1_team2)
    offensive_stats_team1_team2 = offensive_data[offensive_data['Team'].isin([team1, team2])]
    print("Offensive Stats for Team1 and Team2:")
    print(offensive_stats_team1_team2)
    st_stats_team1_team2 = st_data[st_data['Team'].isin([team1, team2])]
    print("ST Stats for Team1 and Team2:")
    print(st_stats_team1_team2)

    go_data = load_game_outcome(go_merged_data_path)
    go_data = clean_game_outcome(go_data)
    go_data = preprocess_game_outcome(go_data)


    #Printing Correlation Matrixes to see which features are strongest
    offensive_numeric_data = offensive_data.select_dtypes(include=[np.number])
    offensive_correlation_matrix = offensive_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(offensive_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Offensive Data Correlation Matrix for {team1} and {team2}")
    plt.show()

    defensive_numeric_data = defensive_data.select_dtypes(include=[np.number])
    defensive_correlation_matrix = defensive_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(defensive_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Defensive Data Correlation Matrix for {team1} and {team2}")
    plt.show()

    st_numeric_data = st_data.select_dtypes(include=[np.number])
    st_correlation_matrix = st_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(st_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"ST Data Correlation Matrix for {team1 and team2}")
    plt.show()

    go_numeric_data = go_data.select_dtypes(include=[np.number])
    go_correlation_matrix = go_numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(go_correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"GO Data Correlation Matrix for {team1} and {team2}")
    plt.show()

    #Multicollinearity
    corr_threshold = 0.8
    def drop_correlated_features(df, threshold):
    # Select only numeric columns for the correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()  # Get the absolute value of correlation coefficients
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Make sure to drop the features from the original dataframe
        df_dropped = df.drop(columns=to_drop, axis=1, errors='ignore')  # Drop features above the threshold
        return df_dropped, to_drop

    offensive_data, dropped_offensive = drop_correlated_features(offensive_stats_team1_team2, corr_threshold)
    defensive_data, dropped_defensive = drop_correlated_features(defensive_stats_team1_team2, corr_threshold)

    # Dimensionality Reduction

    # Concatenate your DataFrames (assuming they're aligned properly)
    combined_df = pd.concat([
        offensive_data.select_dtypes(include=[np.number]),
        defensive_data.select_dtypes(include=[np.number]),
        st_data.select_dtypes(include=[np.number])
    ], axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(combined_df)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imputed)
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X_std)
    print(f"Number of components that explain 95% of the variance: {pca.n_components_}")
    pc_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    
    # DR for Game Outcome
    numeric_go_data = go_data.select_dtypes(include=[np.number])
    go_X_std = StandardScaler().fit_transform(numeric_go_data)
    go_pca = PCA(n_components=0.95)
    go_principalComponents = go_pca.fit_transform(go_X_std)
    print(go_pca.explained_variance_ratio_)
    print(f"Number of components that explain 95% of the variance in Game Outcome data: {go_pca.n_components_}")
    go_pc_df = pd.DataFrame(go_principalComponents, columns=[f'PC{i+1}' for i in range(go_pca.n_components_)])

    #Train the Model
    pc_df.reset_index(drop=True, inplace=True)
    go_data.reset_index(drop=True, inplace=True)

    # Select 500 random rows from the aligned DataFrames
    random_indices = np.random.choice(pc_df.index, size=500, replace=False)  # Ensures alignment
    X_random = pc_df.iloc[random_indices]
    y_random = go_data.iloc[random_indices]['WinningTeam']  # Adjust 'WinningTeam' to your target column in go_data

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_random, y_random, test_size=0.2, random_state=42)
    unique_labels = y_train.unique()
    class_weights = {label: 1.0 for label in unique_labels}
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    # Assuming y_test and y_pred are your test labels and predictions respectively
    print(classification_report(y_test, y_pred, zero_division=0))
    # The accuracy_score function requires actual and predicted labels as inputs
    print(accuracy_score(y_test, y_pred))

    # Return the trained model and the principal components DataFrame
    combined_stats = pc_df

    return rf_model, combined_stats
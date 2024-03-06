import pandas as pd
import os

def merge_csv_in_folder(folder_path, output_file):
    # List to hold data from each CSV file
    all_data = []

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_data.append(df)

    # Concatenate all dataframes
    merged_data = pd.concat(all_data, ignore_index=True)

    # Save to a new CSV file
    merged_data.to_csv(output_file, index=False)

# Usage
folder_path = 'team_stats\Special_team'
output_file = 'STmerged_data.csv'
merge_csv_in_folder(folder_path, output_file)

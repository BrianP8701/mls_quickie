import pandas as pd
import os

# Specify the folder containing the CSV files
folder_path = 'data/'

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Load and concatenate all CSV files into one DataFrame
df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Display the first few rows of the combined DataFrame
print(combined_df.head())

combined_df.to_csv('combined_data.csv', index=False)
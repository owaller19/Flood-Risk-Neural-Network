import pandas as pd
import numpy as np

# Remove any values more than 3 standard deviations away from the mean
def remove_outliers(df):
    z_scores = df.apply(lambda x: np.abs((x - x.mean()) / x.std()))
    return df[(z_scores < 3).all(axis=1)]

# Function to scale data between 0.1 and 0.9
def standardise_data(df):
    min_val = df.min()
    max_val = df.max()
    return 0.1 + (0.9 - 0.1) * (df - min_val) / (max_val - min_val)

file_name = 'preparedata.xlsx'  
sheet_name = 'Sheet1'  

df = pd.read_excel(file_name, sheet_name=sheet_name)
df = remove_outliers(df)

df_scaled = standardise_data(df)
output_file_name = 'standardised_data.xlsx'  
df_scaled.to_excel(output_file_name, index=False)

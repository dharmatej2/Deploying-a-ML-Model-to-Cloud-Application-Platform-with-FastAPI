import pandas as pd

# Read the data
df = pd.read_csv('census.csv')
print("Columns before cleaning:")
print(df.columns.tolist())

# Remove spaces from column names
df.columns = df.columns.str.replace(' ', '')
print("Columns after cleaning:")
print(df.columns.tolist())

# Optionally, save the cleaned data for consistency (if required)
df.to_csv('census_clean.csv', index=False)
print("Cleaned data saved as census_clean.csv")
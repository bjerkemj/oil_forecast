import os
import pandas as pd

# Path to the CSV file to load
csv_filename = "field_production_data/raw_data.csv"

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(csv_filename)

# There seems to be a prefix in the column names, which makes sense to remove
# Define the prefix to remove
prefix_to_remove = "prf"

# Remove the prefix from column names
df.columns = [col.replace(prefix_to_remove, "") for col in df.columns]

# Columns to keep (I've made sure to extract the columns with the data I need, even though the column names are different)
selected_columns = ['InformationCarrier', 'Year', 'Month', 'PrdOilNetMillSm3']
df = df[selected_columns]

# Rename columns
df.columns = ['Field (Discovery)', 'Year', 'Month', 'Net - oil [mill Sm3]']

# Field names to extract
field_names = ['EKOFISK', 'ELDFISK', 'EMBLA']

# Extract rows with the specified field names
df = df[df['Field (Discovery)'].isin(field_names)]

# Filter by year
df = df[df['Year'] >= 2013]

# Time for conversion (these could easily be combined into one line, but just want to show my thinking process)
# Define the conversion factor
conversion_factor = 6.29

# Convert the 'Net - oil [mill Sm3]' to just Sm3
df['Net - oil [Sm3]'] = df['Net - oil [mill Sm3]'] * 1000000

# Convert the 'Net - oil [Sm3]' column to barrels per day
df['Net - oil [bbl/day]'] = df['Net - oil [Sm3]'] * conversion_factor

# Convert the 'Net - oil [bbl/day]' column to thousand barrels per day
df['Net - oil [kbbl/day]'] = df['Net - oil [bbl/day]'] / 1000

# List of columns to drop
columns_to_drop = ['Net - oil [mill Sm3]', 'Net - oil [Sm3]', 'Net - oil [bbl/day]']

# Drop the specified columns
df = df.drop(columns=columns_to_drop)

# Create a 'Date' column based on 'Year' and 'Month' with day set to '01'
# This equals 01/mm/yyyy
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

# Group by 'Date' and sum up production data
grouped_df = df.groupby(['Date']).agg({'Net - oil [kbbl/day]': 'sum'}).reset_index()

# Add a column with the field name
grouped_df['Field (Discovery)'] = 'Greater Ekofisk Area'

# Select desired columns from the original dataframe
df = df[['Field (Discovery)', 'Date', 'Net - oil [kbbl/day]']]

# Concatenate the two dataframes
result_df = pd.concat([df, grouped_df], ignore_index=True)

# Lastly, saving the dataframe
# Define the folder name
folder_name = "field_production_data"

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Specify the path to save the CSV file inside the folder
csv_filename = os.path.join(folder_name, "processed_data.csv")

# Save the dataframe to a CSV file 
result_df.to_csv(csv_filename, index=False)

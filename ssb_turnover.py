import matplotlib.pyplot as plt
import pandas as pd

# Specify the file path to your CSV file
csv_file_path = "ssb_turnover_data.csv"

# Load the CSV file into a Pandas DataFrame using a semicolon as the delimiter
df = pd.read_csv(csv_file_path, delimiter=';')

# Convert the 'Date' column to a proper datetime format
df['Date'] = pd.to_datetime(df['Date'].str[:4] + '-' + df['Date'].str[5:7] + '-01')

# Iterate through the columns and create individual line plots
for column in df.columns[1:]:  # Start from the second column, assuming the first column is 'Date'
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df[column])
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title(f'{column} Time Series')
    plt.grid()
    plt.tight_layout()
    plt.show()

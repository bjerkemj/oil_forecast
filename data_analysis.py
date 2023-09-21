import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Path to the CSV file to load
csv_filename = "field_production_data/processed_data.csv"

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(csv_filename)

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Create subplots for better visualization
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the time series data
for field_name, field_data in df.groupby('Field (Discovery)'):
    ax.plot(field_data['Date'], field_data['Net - oil [kbbl/day]'], label=field_name)

# Set labels, title, legend and styling 
ax.set_xlabel('Date')
ax.set_ylabel('Production (kbbl/day)')
ax.set_title('Production Time Series by Field')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
# plt.show()

# Note to plot: In the month of June 2013 and every three years after that there is a sudden drop in production. Due to the perfect intervals of the drops, it is likely that this is due to maintanence on the fields. Could not find any information on online sources to confirm this.

#Checking if the data is stationary. As the plots suggests unsationary by eye, except for EMBLA. The means and variances suggests unstationary for all fields. An ADF test suggest unstationary for EMBLA and EKOFISK. As the evidence is so varied, I will assume the data is unstationary.
from statsmodels.tsa.stattools import adfuller

field_names = ['EKOFISK', 'ELDFISK', 'EMBLA', 'Greater Ekofisk Area']

for field_name in field_names:
    data = df[df['Field (Discovery)'] == field_name]['Net - oil [kbbl/day]'].values

    # Get the mean and variance of the first and second half of the data
    split = len(data) // 2
    X1, X2 = data[0:split], data[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print(field_name)
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))

    # Perform an ADF test for the data
    result_adf = adfuller(data)
    print('p-value:', result_adf[1])
    print()

# Create a df with only Greater Ekofisk Area data. This will be field in focus for the rest of the analysis due to the short timeframe to complete the task.
df_gea = df[df['Field (Discovery)'] == 'Greater Ekofisk Area'].copy()

# Make the data stationary by performing differencing in first-order
greater_ekofisk_diff = df_gea['Net - oil [kbbl/day]'].diff()

# Plot the differenced data. The plot suggests that the data is stationary now.
plt.figure(figsize=(12, 6))
plt.plot(df_gea['Date'], greater_ekofisk_diff, label='Greater Ekofisk Area (Differenced)', color='blue')
plt.xlabel('Date')
plt.ylabel('Differenced Production (kbbl/day)')
plt.title('Differenced Production Time Series for Greater Ekofisk Area')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
# plt.show()

# Untill now the idea of time series forecast seemed like a good solution. But with the time given I will predict using simpler tools, such as linear regression.
# Firstly i will smooth out the huge production drops in the data. I will do this by taking the mean of the production before and after the drop and use that as the production for the drop.

# Dates with huge production drops
drop_dates = ['2013-06-01', '2016-06-01', '2019-06-01', '2022-06-01']

# Define a list of window sizes to try
window_sizes = [3, 5, 7]

# Create new columns for smoothed production data for each window size
for window_size in window_sizes:
    col_name = f'Smoothed Production (kbbl/day) - Window Size {window_size}'
    df_gea[col_name] = df_gea.loc[:,'Net - oil [kbbl/day]']

# Smooth out production drops for each window size
for date in drop_dates:
    drop_date = pd.to_datetime(date)
    
    for window_size in window_sizes:
        prev_date = drop_date - pd.DateOffset(months=window_size // 2)
        next_date = drop_date + pd.DateOffset(months=window_size // 2)
    
        # Calculate the mean production within the window
        mean_production = df_gea[(df_gea['Date'] >= prev_date) & (df_gea['Date'] <= next_date)]['Net - oil [kbbl/day]'].mean()
    
        # Update the corresponding smoothed production column with the mean
        col_name = f'Smoothed Production (kbbl/day) - Window Size {window_size}'
        df_gea.loc[df_gea['Date'] == drop_date, col_name] = mean_production

# Plot original and smoothed production data for all window sizes
plt.figure(figsize=(12, 6))
plt.plot(df_gea['Date'], df_gea['Net - oil [kbbl/day]'], label='Original Production', color='blue')

for window_size in window_sizes:
    col_name = f'Smoothed Production (kbbl/day) - Window Size {window_size}'
    plt.plot(df_gea['Date'], df_gea[col_name], label=f'Smoothed Production (Window Size {window_size})')

plt.xlabel('Date')
plt.ylabel('Production (kbbl/day)')
plt.title('Original vs. Smoothed Production Time Series for Greater Ekofisk Area')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# By the plot I decide that we will continue on with the smoothed production data with a window size of 7 months.
# Now i update the dataframe with only the data we need.
df_gea = df_gea[['Date', 'Smoothed Production (kbbl/day) - Window Size 7']].copy()
df_gea.columns = ['Date', 'Production']

# Now I bring in some data from SSB that might be usefull. To see some plots of the data check out the file ssb_turnover.py.
# Path to the CSV file to load
csv_filename = "ssb_turnover_data.csv"

# Load the CSV data into a pandas DataFrame
df_ssb = pd.read_csv(csv_filename, delimiter=';')
df_ssb['Date'] = pd.to_datetime(df_ssb['Date'].str[:4] + '-' + df_ssb['Date'].str[5:7] + '-01')
for column in df_ssb.columns:
    if df_ssb[column].dtype == 'object':
        df_ssb[column] = df_ssb[column].str.replace(',', '.')
        df_ssb[column] = pd.to_numeric(df_ssb[column], errors='coerce')

# Merge all the data
df_gea = pd.merge(df_ssb, df_gea, on='Date', how='inner')

# Choose a reference date (starting date)
reference_date = pd.to_datetime('2003-01-01')  # Replace 'your_reference_date_here' with your chosen date

# Calculate the number of days since the reference date
df_gea['Days'] = (df_gea['Date'] - reference_date).dt.days
df_gea = df_gea.drop(columns=['Date'])

from sklearn.model_selection import train_test_split

# Creating train and test data
X = df_gea.drop(columns=['Production'])
y = df_gea['Production']

# Split the data into a random train (80%) and test (20%) set. Random state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Function to perform polynomial regression and return results
def polynomial_regression(X_train, y_train, X_test, y_test, degree):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Calculate RMSE and R-squared
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return {
        'Degree': degree,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R-squared': r2_train,
        'Test R-squared': r2_test,
        'Model': model,
    }

# Perform polynomial regression for degrees 1, 2, and 3
results = []

for degree in [1, 2, 3]:
    result = polynomial_regression(X_train, y_train, X_test, y_test, degree)
    results.append(result)

# Print the results for each degree
for result in results:
    print(f"Degree {result['Degree']} Polynomial Regression:")
    print(f"Train RMSE: {result['Train RMSE']:.2f}")
    print(f"Test RMSE: {result['Test RMSE']:.2f}")
    print(f"Train R-squared: {result['Train R-squared']:.2f}")
    print(f"Test R-squared: {result['Test R-squared']:.2f}")
    print()

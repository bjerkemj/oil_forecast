import os
import requests

# URL for fetching the CSV file
csv_url = "https://factpages.npd.no/ReportServer_npdpublic?/FactPages/tableview/field_production_monthly&rs:Command=Render&rc:Toolbar=false&rc:Parameters=f&IpAddress=not_used&CultureCode=en&rs:Format=CSV&Top100=false"

# Define the folder name
folder_name = "field_production_data"

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Specify the path to save the CSV file inside the folder
csv_filename = os.path.join(folder_name, "raw_data.csv")

# Request to download the CSV file
response = requests.get(csv_url)

# Check if the request was successful
if response.status_code == 200:
    # Save the CSV content to a file in the specified folder
    with open(csv_filename, 'wb') as csv_file:
        csv_file.write(response.content)

    print(f"CSV file '{csv_filename}' downloaded successfully.")
else:
    print("Failed to download the CSV file. Status code:", response.status_code)

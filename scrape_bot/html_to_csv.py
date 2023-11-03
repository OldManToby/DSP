import csv
from bs4 import BeautifulSoup

# Read data from the HTML file
with open("stats_2022.html", "r") as file:
    content = file.read()

soup = BeautifulSoup(content, 'html.parser')
rows = soup.find_all("tr")

# Extract data from each row and store it in a list of lists
data = []
for row in rows:
    columns = row.find_all("td")
    row_data = [col.text.strip() for col in columns]
    data.append(row_data)

# Write the data to a CSV file
with open("stats.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header if available (assuming the first row contains header data)
    if data:
        csvwriter.writerow(data[0])
        data = data[1:]  # Remove the header row from the data
    csvwriter.writerows(data)

print("CSV file has been created successfully.")

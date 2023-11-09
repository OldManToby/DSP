import json
import csv
import subprocess
import os
from bs4 import BeautifulSoup
import re
import datetime

current_year = datetime.datetime.now().year

web_scraper_path = "C:\\Users\\Toby\\Documents\\GitHub\\DSP\\scrape_bot\\web_scraper.py"
config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# Read header mappings from config.json
with open(config_file_path, "r") as config_file:
    header_mapping = json.load(config_file)

for year in range(2000, current_year + 1):
    urls_to_scrape = [
        f"https://www.nfl.com/stats/team-stats/defense/passing/{year}/reg/all",
        f"https://www.nfl.com/stats/team-stats/defense/rushing/{year}/reg/all",
        f"https://www.nfl.com/stats/team-stats/defense/scoring/{year}/reg/all",
        f"https://www.nfl.com/stats/team-stats/defense/downs/{year}/reg/all",
        f"https://www.nfl.com/stats/team-stats/defense/fumbles/{year}/reg/all",
        f"https://www.nfl.com/stats/team-stats/defense/interceptions/{year}/reg/all",
    ]

    # Output directories for HTML and CSV files
    base_output_directory = os.path.join("C:\\Users\\Toby\\Documents\\GitHub\\DSP\\team_stats", str(year), "Defensive")
    os.makedirs(base_output_directory, exist_ok=True)

    def scrape_and_save_html(url, index):
        html_output_filename = f"stats_{index}.html"
        html_output_filepath = os.path.join(base_output_directory, html_output_filename)
        subprocess.run(["python", web_scraper_path, url, html_output_filepath])

    def convert_html_to_csv(index):
        html_input_filename = f"stats_{index}.html"
        html_input_filepath = os.path.join(base_output_directory, html_input_filename)

        with open(html_input_filepath, "r") as file:
            content = file.read()
        cleaned_content = re.sub(r'\s+', ' ', content)
        cleaned_content = re.sub(r'<td>([^<]+)\s+([^<]+)<', r'<td>\1<', cleaned_content)

        soup = BeautifulSoup(cleaned_content, 'html.parser')
        rows = soup.find_all("tr")
        headers = [col.text.strip() for col in rows[0].find_all("td")]
        mapped_headers = [header_mapping.get(header, header) for header in headers]

        data = []
        for row in rows[1:]:
            columns = row.find_all("td")
            row_data = [col.text.strip() for col in columns]
            cleaned_row_data = [item for item in row_data if item]
            data.append(cleaned_row_data)

        csv_output_filename = f"stats_{index}.csv"
        csv_output_filepath = os.path.join(base_output_directory, csv_output_filename)
        with open(csv_output_filepath, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(mapped_headers)
            csvwriter.writerows(data)
        print(f"CSV file {csv_output_filename} has been created successfully.")

    for index, url_to_scrape in enumerate(urls_to_scrape, start=1):
        scrape_and_save_html(url_to_scrape, index)
    for index in range(1, len(urls_to_scrape) + 1):
        convert_html_to_csv(index)

# web_scraper.py

import sys
import requests
from bs4 import BeautifulSoup

def scrape_and_save_data(url, output_filename):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find("table")

    # Extract the table rows
    rows = table.find_all("tr")

    # Extract data from each row and write it to the output file
    with open(output_filename, "w+") as file:
        for row in rows:
            columns = row.find_all(["th", "td"])
            row_data = [col.text.strip() for col in columns]
            file.write("<tr>")
            for data in row_data:
                file.write(f"<td>{data}</td>")
            file.write("</tr>\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python web_scraper.py <URL> <output_filename>")
        sys.exit(1)

    url = sys.argv[1]
    output_filename = sys.argv[2]
    scrape_and_save_data(url, output_filename)

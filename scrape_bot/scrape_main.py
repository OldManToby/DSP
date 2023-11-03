# run_web_scraper.py

import subprocess

# Full path to web_scraper.py
web_scraper_path = "C:\\Users\\Toby\\Documents\\GitHub\\DSP\\scrape_bot\\web_scraper.py"

# URL to scrape
url_to_scrape = "https://www.nfl.com/stats/team-stats/offense/passing/2022/reg/all"

# Output HTML filename
output_filename = "stats_2022.html"

# Run the web scraper with the URL and output filename
subprocess.run(["python", web_scraper_path, url_to_scrape, output_filename])

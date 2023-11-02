import requests
from bs4 import BeautifulSoup


URL = "https://www.nfl.com/stats/team-stats/offense/passing/2022/reg/all"
page = requests.get(URL)

s = BeautifulSoup(page.content, 'html.parser')
results = s.find(id='')
team_stats = results.find_all
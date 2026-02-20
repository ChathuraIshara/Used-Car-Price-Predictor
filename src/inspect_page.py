import cloudscraper
from bs4 import BeautifulSoup

scraper_obj = cloudscraper.create_scraper(browser='chrome')
r = scraper_obj.get('https://riyasewana.com/search/cars', timeout=20)

with open('fresh_dump.html', 'w', encoding='utf-8') as f:
    f.write(r.text)

soup = BeautifulSoup(r.text, 'html.parser')

# Try many selectors
selectors_to_try = [
    ('div.item', soup.select('div.item')),
    ('li.item', soup.select('li.item')),
    ('div[class*=item]', soup.find_all('div', class_=lambda c: c and 'item' in ' '.join(c))),
    ('a[href*=/buy/]', soup.find_all('a', href=lambda h: h and '/buy/' in h)),
    ('h2 a[href*=/buy/]', soup.select('h2 a')),
]

for name, results in selectors_to_try:
    print(f'{name}: {len(results)}')
    if results and len(results) < 10:
        for r2 in results[:3]:
            txt = r2.get_text(strip=True)[:50].encode('ascii','replace').decode()
            print(f'  {txt}')

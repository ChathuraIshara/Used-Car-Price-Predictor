"""
Riyasewana.com Car Listings Scraper
====================================
Scrapes car listings from Riyasewana.com and saves them as a CSV dataset
suitable for machine learning tasks.

Usage:
  python scraper.py                  # Scrape with default settings (50 pages)
  python scraper.py --pages 5        # Scrape only 5 pages
  python scraper.py --pages 5 --output my_data.csv
  python scraper.py --verbose        # Enable debug logging

Dependencies: pip install requests beautifulsoup4 lxml pandas cloudscraper
"""

import argparse
import csv
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional

import cloudscraper
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_URL = "https://riyasewana.com"
SEARCH_URL = "https://riyasewana.com/search/cars"

DEFAULT_MAX_PAGES = 120        # ~42 listings/page × 120 pages ≈ 5040 listings
DEFAULT_DELAY = 1.5          # seconds between requests (be polite)
DEFAULT_OUTPUT = "riyasewana_cars.csv"

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────
@dataclass
class CarListing:
    title: str = ""
    make: str = ""
    model: str = ""
    year: str = ""
    price_lkr: str = ""
    mileage_km: str = ""
    location: str = ""
    fuel_type: str = ""
    transmission: str = ""
    engine_cc: str = ""
    options: str = ""
    details: str = ""
    condition: str = ""
    date_posted: str = ""
    listing_url: str = ""


CSV_COLUMNS = [f.name for f in CarListing.__dataclass_fields__.values()]

# ─────────────────────────────────────────────────────────────
# HTTP SESSION (cloudscraper bypasses Cloudflare)
# ─────────────────────────────────────────────────────────────
SCRAPER = cloudscraper.create_scraper(browser="chrome")


def fetch(url: str, retries: int = 3, backoff: float = 3.0) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object, with retry logic."""
    for attempt in range(1, retries + 1):
        try:
            log.debug("GET %s  (attempt %d)", url, attempt)
            resp = SCRAPER.get(url, timeout=20)
            if resp.status_code == 404:
                log.warning("404 Not Found: %s", url)
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            log.warning("Request failed: %s  (attempt %d/%d)", exc, attempt, retries)
            if attempt < retries:
                time.sleep(backoff * attempt)
    log.error("All retries exhausted for: %s", url)
    return None


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def clean(text: Optional[str]) -> str:
    """Strip whitespace/newlines from scraped text."""
    if not text:
        return ""
    return " ".join(text.split())


def digits_only(text: str) -> str:
    """Extract only digit characters from a string."""
    return re.sub(r"[^\d]", "", text)


def page_url(page: int) -> str:
    """Build the paginated search URL.
    Page 1 → https://riyasewana.com/search/cars
    Page N → https://riyasewana.com/search/cars?page=N
    """
    return SEARCH_URL if page == 1 else f"{SEARCH_URL}?page={page}"


# ─────────────────────────────────────────────────────────────
# LISTING INDEX PAGE  —  extract card-level data
# ─────────────────────────────────────────────────────────────
def scrape_listing_cards(soup: BeautifulSoup) -> list[dict]:
    """
    Extract basic listing data from a Riyasewana search-results page.

    HTML structure observed:
      <div class="item [promoted]">
        <h2 class="more [promoted]">
          <a href="/buy/...">TITLE</a>
        </h2>
        <div class="boxintxt">
          <b>Rs. X,XXX,XXX</b>
          CITY
          X km
          <s>YYYY-MM-DD</s>
        </div>
      </div>
    """
    cards: list[dict] = []
    # Riyasewana serves listing cards as either <div class="item"> or <li class="item">
    # on different requests.  find_all with class_='item' matches both.
    items = soup.find_all(True, class_="item")
    log.debug("Found %d .item cards on page", len(items))

    # Regex for date: YYYY-MM-DD
    date_pat = re.compile(r'\d{4}-\d{2}-\d{2}')

    for item in items:
        try:
            #── Title & URL ──────────────────────────────────────────
            # <h2 class="more"> <a href="..."> ... </a> </h2>
            title_tag = item.find("h2").find("a")
            title = title_tag.get_text(strip=True)
            link = title_tag["href"]

            # Fix relative URLs if any (Riyasewana usually uses absolute, but safe to check)
            if link and link.startswith("/"):
                link = BASE_URL + link

            #── Info Boxes (Price, Location, Mileage, Date) ──────
            # <div class="boxintxt">...</div> matches multiple times
            # 1. Location (usually 1st)
            # 2. Price (usually 2nd, contains "Rs." or "Negotiable")
            # 3. Mileage (contains "km")
            # 4. Date (YYYY-MM-DD, sometimes inside <s>)
            
            box_divs = item.find_all("div", class_="boxintxt")
            
            price_str = ""
            loc_str = ""
            mileage_str = ""
            date_str = ""
            
            # Helper to check if string looks like a date
            def is_date(s): return bool(date_pat.search(s))

            for box in box_divs:
                txt = box.get_text(strip=True)
                low = txt.lower()
                
                if "rs." in low or "negotiable" in low:
                    price_str = txt
                elif "(km)" in low:
                    mileage_str = txt
                elif is_date(txt):
                    date_str = txt
                else:
                    # Assume location if not any of the above
                    # (First non-matching div is usually location)
                    if not loc_str: 
                        loc_str = txt

            # Fallback for date if not found in text (check <s> tag)
            if not date_str:
                s_tag = item.find("s")
                if s_tag:
                    date_str = s_tag.get_text(strip=True)

            #── Clean & Store ────────────────────────────────────────
            cards.append({
                "title":       title,
                "listing_url": link,
                "price_lkr":   digits_only(price_str),
                "location":    clean(loc_str),
                "mileage_km":  digits_only(mileage_str),
                "date_posted": clean(date_str)
            })

        except Exception as exc:
            log.debug("Error parsing card: %s", exc)
            continue

    return cards


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL LISTING PAGE  —  extract detailed specs
# ─────────────────────────────────────────────────────────────
# Map label text (lowercase) → CarListing field name
LABEL_MAP = {
    "make":         "make",
    "brand":        "make",
    "model":        "model",
    "yom":          "year",
    "year":         "year",
    "manufactured": "year",
    "mileage":      "mileage_km",
    "distance":     "mileage_km",
    "fuel type":    "fuel_type",
    "fuel":         "fuel_type",
    "gear":         "transmission",
    "transmission": "transmission",
    "engine":       "engine_cc",
    "cc":           "engine_cc",
    "options":      "options",
    "details":      "details",
    "condition":    "condition",
}


def match_label(label: str) -> Optional[str]:
    """Return the CarListing field name for a given spec label, or None."""
    label_lower = label.lower().strip()
    for key, field in LABEL_MAP.items():
        if key in label_lower:
            return field
    return None


def scrape_listing_detail(url: str, delay: float) -> dict:
    """
    Visit an individual listing page and extract detailed car specs.

    Detail page HTML (table.moret):
      All spec cells share class 'aleft'. They appear in label/value pairs
      in document order. CSS selectors fail with lxml for multi-class elements;
      use find_all(class_=) instead which does partial class matching.
    """
    time.sleep(delay)
    soup = fetch(url)
    if not soup:
        return {}

    details: dict = {}

    # find_all with class_='aleft' correctly matches elements that CONTAIN
    # the 'aleft' class (even alongside other classes like 'ftin', 'tfiv').
    # This is NOT equivalent to soup.select('td.aleft') when lxml is used.
    spec_tds = soup.find_all("td", class_="aleft")

    # Pair spec TDs in order: [label, value, label, value, ...]
    for i in range(0, len(spec_tds) - 1, 2):
        label = clean(spec_tds[i].get_text())
        value = clean(spec_tds[i + 1].get_text())
        field = match_label(label)
        if field and value:
            details.setdefault(field, value)

    # Normalise mileage to digits only
    if "mileage_km" in details:
        details["mileage_km"] = digits_only(details["mileage_km"])

    # Normalise engine cc to digits only
    if "engine_cc" in details:
        details["engine_cc"] = digits_only(details["engine_cc"])

    return details


# ─────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────
def scrape(max_pages: int, delay: float, output_file: str,
           start_page: int = 1, append: bool = False) -> None:
    log.info("=" * 60)
    log.info("Riyasewana.com Car Scraper")
    log.info("  Pages  : %d-%d", start_page, max_pages)
    log.info("  Delay  : %.1f s", delay)
    log.info("  Output : %s (%s)", output_file, 'append' if append else 'overwrite')
    log.info("=" * 60)

    total_scraped = 0
    file_mode = "a" if append else "w"

    with open(output_file, file_mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not append:
            writer.writeheader()

        for page_num in range(start_page, max_pages + 1):
            url = page_url(page_num)
            log.info("[Page %d/%d] %s", page_num, max_pages, url)

            soup = fetch(url)
            if not soup:
                log.warning("Could not fetch page %d. Stopping.", page_num)
                break

            cards = scrape_listing_cards(soup)
            if not cards:
                log.info("No listings found on page %d. Reached end of results.", page_num)
                break

            log.info("  \u2192 %d listings found", len(cards))

            for idx, card in enumerate(cards, 1):
                listing = CarListing(
                    title       = card.get("title", ""),
                    price_lkr   = card.get("price_lkr", ""),
                    mileage_km  = card.get("mileage_km", ""),
                    location    = card.get("location", ""),
                    date_posted = card.get("date_posted", ""),
                    listing_url = card.get("listing_url", ""),
                )

                detail_url = card.get("listing_url", "")
                if detail_url:
                    log.debug("    [%d/%d] %s", idx, len(cards), detail_url)
                    det = scrape_listing_detail(detail_url, delay)
                    listing.make         = det.get("make", "")
                    listing.model        = det.get("model", "")
                    listing.year         = det.get("year", "")
                    listing.fuel_type    = det.get("fuel_type", "")
                    listing.transmission = det.get("transmission", "")
                    listing.engine_cc    = det.get("engine_cc", "")
                    listing.options      = det.get("options", "")
                    listing.details      = det.get("details", "")
                    listing.condition    = det.get("condition", "")
                    if det.get("mileage_km"):
                        listing.mileage_km = det["mileage_km"]

                writer.writerow(asdict(listing))
                csvfile.flush()
                total_scraped += 1

                time.sleep(delay)

            log.info("  \u2713 Page %d complete  (newly scraped: %d)", page_num, total_scraped)
            time.sleep(delay)

    log.info("=" * 60)
    log.info("\u2705 Scraping complete!")
    log.info("   Newly scraped : %d", total_scraped)
    log.info("   CSV saved to  : %s", output_file)
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape car listings from Riyasewana.com and save to CSV."
    )
    parser.add_argument(
        "--pages", type=int, default=DEFAULT_MAX_PAGES,
        help=f"Maximum pages to scrape (default: {DEFAULT_MAX_PAGES})"
    )
    parser.add_argument(
        "--start-page", type=int, default=1,
        help="Page number to start from, useful for resuming (default: 1)"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between requests (default: {DEFAULT_DELAY})"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing CSV instead of overwriting"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scrape(
        max_pages=args.pages,
        start_page=args.start_page,
        delay=args.delay,
        output_file=args.output,
        append=args.append,
    )


if __name__ == "__main__":
    main()

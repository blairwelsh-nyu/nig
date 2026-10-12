"""
Scrape Nigeria Watch event database for study period 1999-2016.
Phase 1: Scrape all list pages -> event_id, title, dates, deaths
Phase 2: Scrape detail pages for kidnap/abduct events -> LGA, place, protagonists, causes
"""
import re
import time
import csv
import sys
import random
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl

BASE = "https://www.nigeriawatch.org/index.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research/1.0; academic data collection)",
    "Accept": "text/html,application/xhtml+xml",
}

SEARCH_PARAMS_BASE = {
    "F_evt_form": "1",
    "tri": " ORDER BY  EVT_DATE_DEBUT DESC ",
    "chFields[4]": "EVT_DATE_DEBUT",
    "chTypes[4]": "D",
    "chOps[4]": "AND",
    "chValues[4]": "1999/01/01",
    "chValues2[4]": "2016/12/31",
    "cherche": "1",
    "commande": "",
    "ligne": "",
    "nbLignes": "50",
}

KIDNAP_RE = re.compile(
    r'kidnap|abduct|ransom|hostage|seized|seized.*terrorist|'
    r'captiv|taken hostage|held hostage|forced disappear',
    re.I
)


def fetch(url, post_data=None, retries=4):
    delay = 2
    for attempt in range(retries):
        try:
            data = urlencode(post_data).encode() if post_data else None
            req = Request(url, data=data, headers=HEADERS)
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode('utf-8', errors='replace')
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay + random.random())
            delay *= 2


def parse_list_page(html):
    """Extract rows from a list page. Returns list of dicts."""
    rows = []
    # Each trBody row has two <a> tags with the same href (one for id, one for title)
    pattern = re.compile(
        r'<tr class="trBody">(.*?)</tr>', re.S
    )
    td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.S)
    a_pattern = re.compile(r'href="[^"]*id_evt=(\d+)[^"]*"[^>]*>([^<]+)</a>', re.S)

    for m in pattern.finditer(html):
        row_html = m.group(1)
        tds = [re.sub(r'<[^>]+>', '', td).strip()
               for td in td_pattern.findall(row_html)]
        a_matches = a_pattern.findall(row_html)
        if len(tds) >= 5 and a_matches:
            evt_id = a_matches[0][0]
            identifier = tds[0]
            title = tds[1]
            start_date = tds[2]
            end_date = tds[3]
            deaths = tds[4]
            rows.append({
                'evt_id': evt_id,
                'identifier': identifier,
                'title': title,
                'start_date': start_date,
                'end_date': end_date,
                'deaths': deaths,
            })
    return rows


def fetch_list_page(page_num):
    params = dict(SEARCH_PARAMS_BASE)
    params['page'] = str(page_num)
    html = fetch(f"{BASE}?urlaction=evtList&id=&", post_data=params)
    return page_num, parse_list_page(html)


def parse_detail_page(html):
    """Extract detail fields from an event page."""
    result = {}
    pattern = re.compile(
        r'<td class="label_champs_form"[^>]*>\s*(.*?)\s*</td>\s*'
        r'<td class="val_champs_form">(.*?)</td>',
        re.S
    )
    for m in pattern.finditer(html):
        label = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        value = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        value = re.sub(r'\s+', ' ', value)
        result[label] = value
    return result


def fetch_detail(evt_id):
    url = f"{BASE}?urlaction=evtView&id_evt={evt_id}"
    try:
        html = fetch(url)
        d = parse_detail_page(html)
        return evt_id, d
    except Exception as e:
        return evt_id, {'_error': str(e)}


# ─── Phase 1: scrape list pages ─────────────────────────────────────────────

def scrape_list(max_pages=433, workers=8, out_path='/home/user/nig/nigeria_watch_list.csv'):
    print(f"Phase 1: scraping {max_pages} list pages with {workers} workers...")
    fieldnames = ['evt_id', 'identifier', 'title', 'start_date', 'end_date', 'deaths']
    all_rows = []

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_list_page, p): p
                       for p in range(1, max_pages + 1)}
            completed = 0
            for fut in as_completed(futures):
                page_num, rows = fut.result()
                writer.writerows(rows)
                all_rows.extend(rows)
                completed += 1
                if completed % 50 == 0:
                    print(f"  {completed}/{max_pages} pages done, {len(all_rows)} events so far")

    print(f"Phase 1 complete: {len(all_rows)} events -> {out_path}")
    return all_rows


# ─── Phase 2: scrape detail pages for kidnap events ─────────────────────────

def scrape_details(rows, workers=10, out_path='/home/user/nig/nigeria_watch_details.csv'):
    # Filter to kidnap-related by title
    kidnap_rows = [r for r in rows if KIDNAP_RE.search(r['title'])]
    print(f"\nPhase 2: {len(kidnap_rows)} kidnap-related events to detail-scrape")

    detail_fields = [
        'Identifier', 'Title', 'Description', 'Start date', 'End date',
        'Duration', 'Place', 'Number of death(s)',
        'Number of death(s) according to police', 'Number of death(s) according to press',
        'Protagonists', 'Causes of violence', 'Sources of information', 'LGA',
    ]
    out_fieldnames = ['evt_id', 'start_date', 'end_date', 'deaths'] + detail_fields

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction='ignore')
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_detail, r['evt_id']): r
                       for r in kidnap_rows}
            completed = 0
            for fut in as_completed(futures):
                row_meta = futures[fut]
                evt_id, detail = fut.result()
                out_row = {
                    'evt_id': evt_id,
                    'start_date': row_meta['start_date'],
                    'end_date': row_meta['end_date'],
                    'deaths': row_meta['deaths'],
                    **detail,
                }
                writer.writerow(out_row)
                completed += 1
                if completed % 100 == 0:
                    print(f"  {completed}/{len(kidnap_rows)} detail pages done")

    print(f"Phase 2 complete: {len(kidnap_rows)} events -> {out_path}")


if __name__ == '__main__':
    # Phase 1
    rows = scrape_list(max_pages=433, workers=8)

    # Phase 2
    scrape_details(rows, workers=10)

    print("\nDone.")

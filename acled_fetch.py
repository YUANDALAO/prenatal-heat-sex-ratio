"""Download ACLED event-level data for the 56 primary-analysis countries (all years available) into data/acled/<ISO3>.csv.
Needs data/.acled_token.json from acled_token.py (refreshed automatically with the refresh token when expired).
Filter by ISO 3166 numeric code (avoids country-name mismatches); paged with limit/page until a short page is returned.
"""
import io, json, os, sys, time, urllib.parse, urllib.request
import pandas as pd, pycountry

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-script'   # acleddata.com returns 403 to the default Python-urllib agent
HERE = os.path.dirname(os.path.abspath(__file__)); TOK = os.path.join(HERE, 'data', '.acled_token.json')
OUT = os.path.join(HERE, 'data', 'acled'); os.makedirs(OUT, exist_ok=True)
FIELDS = 'event_id_cnty|event_date|year|time_precision|event_type|sub_event_type|country|iso|admin1|latitude|longitude|geo_precision|fatalities'
LIMIT = 5000


def token():
    t = json.load(open(TOK))
    if time.time() - t['obtained'] > t['expires_in'] - 600:
        form = dict(refresh_token=t['refresh_token'], grant_type='refresh_token', client_id='acled')
        r = urllib.request.Request('https://acleddata.com/oauth/token', data=urllib.parse.urlencode(form).encode(),
                                   headers={'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': UA})
        n = json.load(urllib.request.urlopen(r, timeout=60)); n['obtained'] = time.time()
        t = {k: n.get(k, t.get(k)) for k in ['access_token', 'refresh_token', 'expires_in', 'obtained']}; json.dump(t, open(TOK, 'w'))
    return t['access_token']


def get(params):
    url = 'https://acleddata.com/api/acled/read?' + urllib.parse.urlencode(params, safe='|')
    for attempt in range(5):
        try:
            r = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token(), 'User-Agent': UA})
            return urllib.request.urlopen(r, timeout=300).read().decode('utf-8')
        except Exception as e:
            if attempt == 4: raise
            print('  retry', attempt + 1, e, flush=True); time.sleep(10 * (attempt + 1))


T = pd.read_csv(os.path.join(HERE, 'results', 'country_table.csv'))
for iso3 in T.iso3:
    f = os.path.join(OUT, iso3 + '.csv')
    if os.path.exists(f): continue
    num = int(pycountry.countries.get(alpha_3=iso3).numeric); pages = []; p = 1
    while True:
        txt = get(dict(_format='csv', iso=num, fields=FIELDS, limit=LIMIT, page=p))
        d = pd.read_csv(io.StringIO(txt)) if txt.strip() else pd.DataFrame()
        if p == 1 and 'event_date' not in d.columns:
            sys.exit(f'unexpected response for {iso3}: {txt[:300]}')
        pages.append(d)
        if len(d) < LIMIT: break
        p += 1
    D = pd.concat(pages, ignore_index=True)
    D.to_csv(f, index=False)
    print(iso3, len(D), 'events', D.year.min() if len(D) else '-', '-', D.year.max() if len(D) else '-', flush=True)
print('done')

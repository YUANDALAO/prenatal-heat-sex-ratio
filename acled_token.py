"""Obtain an ACLED OAuth token. Run in a terminal; the password is entered at a hidden prompt and is not stored.
Writes data/.acled_token.json (access token 24 h, refresh token 14 days). Do not share or commit that file.
Docs: https://acleddata.com/api-documentation/getting-started
"""
import getpass, json, os, time, urllib.error, urllib.parse, urllib.request

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-script'   # acleddata.com returns 403 to the default Python-urllib agent
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'data', '.acled_token.json')
email = input('myACLED email: ').strip()
form = dict(username=email, password=getpass.getpass('myACLED password (hidden): '),
            grant_type='password', client_id='acled', scope='authenticated')
req = urllib.request.Request('https://acleddata.com/oauth/token', data=urllib.parse.urlencode(form).encode(),
                             headers={'Content-Type': 'application/x-www-form-urlencoded', 'User-Agent': UA})
try:
    tok = json.load(urllib.request.urlopen(req, timeout=60)); tok['obtained'] = time.time()
except urllib.error.HTTPError as e:
    raise SystemExit(f'HTTP {e.code}: {e.read().decode(errors="replace")[:500]}')
json.dump({k: tok[k] for k in ['access_token', 'refresh_token', 'expires_in', 'obtained']}, open(OUT, 'w'))
print('token saved to', OUT, '- valid 24 h (refresh 14 days)')

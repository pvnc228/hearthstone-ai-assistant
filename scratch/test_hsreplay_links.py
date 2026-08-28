import urllib.request
import re

url = 'https://hsreplay.net/uploads/upload/NM2diXLpFvP9aivpjqS3Ag/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        html = resp.read().decode('utf-8')
        print('HTML length:', len(html))
        # Look for replay / game links or redirects
        print('Title/Meta:')
        for line in html.splitlines():
            if '<title>' in line or 'canonical' in line or 'location' in line.lower() or 'game' in line.lower() or 'replay' in line.lower():
                if len(line.strip()) < 200:
                    print(' ', line.strip())
except Exception as e:
    print('Error:', e)

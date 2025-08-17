import requests
import time
import re

PROXY_FILE = "proxies.txt"
OUTPUT_FILE = "working_proxies.txt"
TEST_URL = "https://youtube.com"
TIMEOUT = 7  # seconds

def parse_proxy_line(line):
    """
    Parse a line like:
    Vietnam 1213ms http elite 183.80.22.67:16000
    and return proxy URL like "http://183.80.22.67:16000" and the raw ip:port string
    """
    match = re.search(r'(\d+\.\d+\.\d+\.\d+:\d+)$', line.strip())
    if not match:
        return None, None
    ip_port = match.group(1)
    protocol = "http"
    proxy_url = f"{protocol}://{ip_port}"
    return proxy_url, ip_port

def test_proxy(proxy_url):
    start = time.time()
    try:
        proxies = {
            "http": proxy_url,
            "https": proxy_url,
        }
        response = requests.get(TEST_URL, proxies=proxies, timeout=TIMEOUT)
        delay = time.time() - start
        if response.status_code == 200:
            print(f"WORKING (HTTPS): {proxy_url} - {delay:.2f}s")
            return True, delay
        else:
            print(f"FAILED: {proxy_url} - Status code {response.status_code}")
            return False, None
    except Exception as e:
        print(f"FAILED: {proxy_url} - {type(e).__name__}: {e}")
        return False, None

def load_proxies(filename):
    with open(filename, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    proxies = []
    for line in raw_lines:
        proxy_url, ip_port = parse_proxy_line(line)
        if proxy_url:
            proxies.append((proxy_url, ip_port))
    return proxies

def save_working_proxies(filename, proxies):
    with open(filename, "w") as f:
        for ip_port in proxies:
            f.write(ip_port + "\n")

def main():
    proxies = load_proxies(PROXY_FILE)
    print(f"Testing {len(proxies)} proxies for HTTPS support on {TEST_URL}...")
    working = []
    for proxy_url, ip_port in proxies:
        ok, delay = test_proxy(proxy_url)
        if ok:
            working.append(ip_port)
    print(f"\n=== {len(working)} WORKING PROXIES ===")
    for p in working:
        print(p)
    save_working_proxies(OUTPUT_FILE, working)
    print(f"\nWorking proxies saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

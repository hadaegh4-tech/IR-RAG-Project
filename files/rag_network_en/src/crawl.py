import json, time, re
import requests
from trafilatura import extract
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse, urljoin

HEADERS = {"User-Agent": "IR-RAG-Networking-Project/1.0 (+course project)"}
REQUEST_DELAY_SEC = 1.0
TIMEOUT_SEC = 25

# سقف خزش برای هر دامنه (سریع و امن)
MAX_PAGES_PER_DOMAIN = {
    "en.wikipedia.org": 45,
    "developer.mozilla.org": 35,   # MDN
    "www.cloudflare.com": 25,      # Cloudflare Learning Center
}

ALLOWED_DOMAINS = set(MAX_PAGES_PER_DOMAIN.keys())

def normalize_url(url: str) -> str:
    url = url.split("#", 1)[0]
    if url.endswith("/"):
        url = url[:-1]
    return url

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.text

def get_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return (soup.title.get_text(strip=True) if soup.title else "")[:200]

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = urljoin(base_url, href)
        abs_url = normalize_url(abs_url)
        if abs_url.startswith("http://") or abs_url.startswith("https://"):
            links.append(abs_url)
    return links

def looks_like_content(domain: str, url: str) -> bool:
    # فایل‌های غیرمتنی
    if re.search(r"\.(pdf|zip|rar|jpg|png|gif|mp3|mp4)$", url, re.IGNORECASE):
        return False

    # ویکی‌پدیا: فقط مقاله‌ها
    if domain == "en.wikipedia.org":
        if "/wiki/" not in url:
            return False
        bad = ["/wiki/Special:", "/wiki/Help:", "/wiki/Category:", "/wiki/Template:", "/wiki/Talk:"]
        if any(b in url for b in bad):
            return False

    # MDN: فقط docs
    if domain == "developer.mozilla.org":
        if "/en-US/docs/" not in url:
            return False

    # Cloudflare learning center: فقط learning
    if domain == "www.cloudflare.com":
        if "/learning/" not in url:
            return False
        if "?" in url:
            return False

    return True

def main():
    seed_urls = [
        # Wikipedia (Networking core)
        "https://en.wikipedia.org/wiki/Computer_network",
        "https://en.wikipedia.org/wiki/Internet_protocol_suite",
        "https://en.wikipedia.org/wiki/Transmission_Control_Protocol",
        "https://en.wikipedia.org/wiki/User_Datagram_Protocol",
        "https://en.wikipedia.org/wiki/Domain_Name_System",
        "https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol",
        "https://en.wikipedia.org/wiki/HTTPS",
        "https://en.wikipedia.org/wiki/Transport_layer",
        "https://en.wikipedia.org/wiki/IP_address",
        "https://en.wikipedia.org/wiki/Router_(computing)",
        "https://en.wikipedia.org/wiki/Network_address_translation",
        "https://en.wikipedia.org/wiki/Firewall_(computing)",
        "https://en.wikipedia.org/wiki/Virtual_private_network",

        # MDN (HTTP basics)
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview",
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods",
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status",

        # Cloudflare Learning
        "https://www.cloudflare.com/learning/dns/what-is-dns/",
        "https://www.cloudflare.com/learning/ssl/what-is-https/",
        "https://www.cloudflare.com/learning/network-layer/what-is-a-proxy/",
        "https://www.cloudflare.com/learning/ddos/what-is-a-ddos-attack/",
    ]

    queue = [normalize_url(u) for u in seed_urls]
    seen = set()
    crawled = {d: 0 for d in ALLOWED_DOMAINS}

    out_path = "data/raw_pages.jsonl"
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=sum(MAX_PAGES_PER_DOMAIN.values()))
        while queue:
            url = normalize_url(queue.pop(0))
            if url in seen:
                continue
            seen.add(url)

            domain = urlparse(url).netloc
            if domain not in ALLOWED_DOMAINS:
                continue
            if crawled[domain] >= MAX_PAGES_PER_DOMAIN[domain]:
                continue
            if not looks_like_content(domain, url):
                continue

            try:
                html = fetch(url)
                title = get_title(html)
                text = (extract(html) or "").strip()

                if text:
                    f.write(json.dumps({
                        "url": url,
                        "title": title,
                        "domain": domain,
                        "text": text
                    }, ensure_ascii=False) + "\n")
                    written += 1

                crawled[domain] += 1
                pbar.update(1)

                # ادامه خزش فقط داخل همان دامنه
                for link in extract_links(html, url):
                    if urlparse(link).netloc == domain and link not in seen:
                        if looks_like_content(domain, link):
                            queue.append(link)

                time.sleep(REQUEST_DELAY_SEC)

            except Exception as e:
                print("FAILED:", url, e)

        pbar.close()

    print("DONE:", out_path)
    print("Written:", written)
    print("Crawled:", crawled)

if __name__ == "__main__":
    main()

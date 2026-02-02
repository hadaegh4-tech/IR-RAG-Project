import json, time, re
import requests
from trafilatura import extract
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse, urljoin

HEADERS = {"User-Agent": "IR-RAG-Literature-Project/1.0 (+course project)"}

# تنظیمات خزش مودبانه
REQUEST_DELAY_SEC = 1.5
TIMEOUT_SEC = 20

# سقف تعداد صفحات برای هر دامنه (برای اینکه خزش سنگین نشود)
MAX_PAGES_PER_DOMAIN = {
    "fa.wikipedia.org": 40,
    "ganjoor.net": 60,
}

# فقط همین دامنه‌ها را اجازه بده
ALLOWED_DOMAINS = set(MAX_PAGES_PER_DOMAIN.keys())

def normalize_url(url: str) -> str:
    # حذف fragment (#...)
    url = url.split("#", 1)[0]
    # حذف اسلش اضافی آخر
    if url.endswith("/"):
        url = url[:-1]
    return url

def get_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return (soup.title.get_text(strip=True) if soup.title else "")[:200]

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.text

def extract_links(html: str, base_url: str) -> list[str]:
    """لینک‌های داخل صفحه را استخراج می‌کند و absolute می‌کند."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        # جلوگیری از لینک‌های javascript/mailto
        if href.startswith("javascript:") or href.startswith("mailto:"):
            continue

        abs_url = urljoin(base_url, href)
        abs_url = normalize_url(abs_url)

        # فقط http/https
        if not (abs_url.startswith("http://") or abs_url.startswith("https://")):
            continue

        links.append(abs_url)
    return links

def looks_like_content_url(domain: str, url: str) -> bool:
    """
    فیلتر ساده برای اینکه لینک‌های بی‌ربط/خاص (مثل فایل‌ها) را نخزیم.
    """
    # حذف فایل‌های غیر HTML
    if re.search(r"\.(pdf|zip|rar|jpg|png|gif|mp3|mp4)$", url, re.IGNORECASE):
        return False

    # فیلترهای مخصوص ویکی‌پدیا: فقط صفحات مقاله، نه Special/Help و...
    if domain == "fa.wikipedia.org":
        if "/wiki/" not in url:
            return False
        bad_prefixes = ["/wiki/Special:", "/wiki/Help:", "/wiki/Category:", "/wiki/Template:"]
        for bp in bad_prefixes:
            if bp in url:
                return False

    # فیلترهای ساده برای گنجور: صفحات اصلی شعرها/شاعران معمولاً همین ساختار را دارند
    if domain == "ganjoor.net":
        # خیلی سخت‌گیر نباشیم، ولی لینک‌های خروجی یا search و... را کم کنیم
        if "?" in url:
            return False

    return True

def main():
    # Seed URLها (شروع خزش)
    seed_urls = [
        # ویکی‌پدیا: تعریف‌ها و معرفی شاعران/سبک‌ها
        "https://fa.wikipedia.org/wiki/ادبیات_فارسی",
        "https://fa.wikipedia.org/wiki/حافظ",
        "https://fa.wikipedia.org/wiki/سعدی",
        "https://fa.wikipedia.org/wiki/مولوی",
        "https://fa.wikipedia.org/wiki/فردوسی",
        "https://fa.wikipedia.org/wiki/شاهنامه",
        "https://fa.wikipedia.org/wiki/بوستان",
        "https://fa.wikipedia.org/wiki/گلستان",
        "https://fa.wikipedia.org/wiki/غزل",
        "https://fa.wikipedia.org/wiki/مثنوی",
        "https://fa.wikipedia.org/wiki/رباعی",
        "https://fa.wikipedia.org/wiki/سبک_خراسانی",
        "https://fa.wikipedia.org/wiki/سبک_عراقی",

        # گنجور: چند نقطه شروع (شاعر و مجموعه‌ها)
        "https://ganjoor.net/",
        "https://ganjoor.net/hafez/",
        "https://ganjoor.net/saadi/",
        "https://ganjoor.net/moulavi/",
        "https://ganjoor.net/ferdousi/",
    ]

    # صف خزش
    queue = [normalize_url(u) for u in seed_urls]
    seen = set()

    # شمارنده برای هر دامنه
    crawled_count = {d: 0 for d in ALLOWED_DOMAINS}

    out_path = "data/raw_pages.jsonl"
    written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=sum(MAX_PAGES_PER_DOMAIN.values()))
        while queue:
            url = queue.pop(0)
            url = normalize_url(url)

            if url in seen:
                continue
            seen.add(url)

            domain = urlparse(url).netloc
            if domain not in ALLOWED_DOMAINS:
                continue

            # سقف دامنه
            if crawled_count[domain] >= MAX_PAGES_PER_DOMAIN[domain]:
                continue

            try:
                html = fetch(url)
                title = get_title(html)

                text = extract(html) or ""
                text = text.strip()
                if text:
                    f.write(json.dumps({
                        "url": url,
                        "title": title,
                        "domain": domain,
                        "text": text
                    }, ensure_ascii=False) + "\n")
                    written += 1

                crawled_count[domain] += 1
                pbar.update(1)

                # لینک‌های جدید برای ادامه خزش
                links = extract_links(html, url)
                for link in links:
                    link_domain = urlparse(link).netloc
                    if link_domain == domain and looks_like_content_url(domain, link):
                        if link not in seen:
                            queue.append(link)

                time.sleep(REQUEST_DELAY_SEC)

            except Exception as e:
                # خطا را فقط چاپ کن و ادامه بده
                print("FAILED:", url, e)

        pbar.close()

    print("DONE:", out_path)
    print("Written records:", written)
    print("Crawled counts:", crawled_count)

if __name__ == "__main__":
    main()

"""Collect a bounded, source-attributed public-document snapshot for a RAG demo."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PyPdfError
import requests

ROOT = Path(__file__).resolve().parent
USER_AGENT = "TourismFinanceResearchBot/0.1 (educational public-document snapshot)"
KEYWORDS = re.compile(
    r"turizm|konaklama|seyahat|acenta|acente|kefalet|kredi|tesvik|teşvik|"
    r"destek|finansman|mevzuat|kanun|yonetmelik|yönetmelik|vergi|rehber|"
    r"yatirim|yatırım|isveren|işveren|kurul.karar|faiz|limit|ihracat",
    re.I,
)
ERROR_PAGE = re.compile(
    r"sayfa bulunamad[ıi]|arad[ıi][ğg][ıi]n[ıi]z sayfa|"
    r"access denied|just a moment|request rejected|403 forbidden|404 not found",
    re.I,
)


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_html(data):
    # Some public sites declare legacy encodings while sending valid UTF-8.
    try:
        return BeautifulSoup(data.decode("utf-8-sig"), "html.parser")
    except UnicodeDecodeError:
        return BeautifulSoup(data, "html.parser")


def html_text(soup):
    main = (
        soup.find("main") or soup.find("article")
        or soup.select_one("#icerik, #content-detail, .content-detail, .detail-content")
        or soup.body or soup
    )
    for tag in main.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", main.get_text("\n", strip=True))


def fetch(url, max_bytes=30_000_000):
    # Only follow same-site redirects from the curated public source list.
    current = url
    original_host = urlsplit(url).hostname.removeprefix("www.")
    for _ in range(8):
        parts = urlsplit(current)
        if parts.scheme != "https" or not parts.hostname or parts.username:
            raise ValueError(f"Unsupported URL: {current}")
        if parts.hostname.removeprefix("www.") != original_host:
            raise ValueError(f"Cross-site redirect requires manual review: {current}")
        with requests.get(
            current, timeout=(8, 22), stream=True, allow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        ) as response:
            if response.is_redirect:
                current = urljoin(current, response.headers["Location"])
                continue
            response.raise_for_status()
            chunks, size = [], 0
            for block in response.iter_content(65536):
                size += len(block)
                if size > max_bytes:
                    raise ValueError(f"Document exceeds {max_bytes} bytes")
                chunks.append(block)
            return b"".join(chunks), current, response.headers.get("Content-Type", "")
    raise ValueError("Too many redirects")


def robots_for(url):
    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    parser = RobotFileParser()
    try:
        data, _, _ = fetch(robots_url, 1_000_000)
        parser.parse(data.decode("utf-8", errors="replace").splitlines())
        return parser, None
    except requests.HTTPError as exc:
        if exc.response.status_code == 404:
            parser.parse([])
            return parser, "robots.txt not present (404)"
        return None, f"robots.txt unavailable: {exc}"
    except (requests.RequestException, ValueError) as exc:
        return None, f"robots.txt unavailable: {exc}"


def collect_one(source, output, robots):
    entry = dict(source)
    entry["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    # Retrieval time is not a statement about legal validity or program availability.
    entry["validity_status"] = "not_verified"
    parser, notice = robots
    entry["robots_notice"] = notice
    if parser is None or not parser.can_fetch(USER_AGENT, source["url"]):
        entry.update(status="blocked", error=notice or "Disallowed by robots.txt")
        return entry, []
    try:
        data, final_url, content_type = fetch(source["url"])
        if not parser.can_fetch(USER_AGENT, final_url):
            raise ValueError("Final URL disallowed by robots.txt")
        entry.update(final_url=final_url, content_type=content_type)
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        pages, links = [], []
        if data.startswith(b"%PDF-"):
            reader = PdfReader(io.BytesIO(data))
            for number, page in enumerate(reader.pages, 1):
                pages.append({"page": number, "text": page.extract_text() or ""})
            suffix = ".pdf"
            entry["format"] = "pdf"
            entry["empty_pages"] = [p["page"] for p in pages if not p["text"].strip()]
            entry["extraction_warning"] = (
                "Some pages require OCR or are blank" if entry["empty_pages"] else None
            )
        else:
            if "pdf" in content_type.lower() or urlsplit(final_url).path.lower().endswith(".pdf"):
                raise ValueError("Expected PDF but server returned non-PDF data")
            if "html" not in content_type.lower() and not data.lstrip().startswith(b"<"):
                raise ValueError(f"Unsupported content type: {content_type}")
            soup = parse_html(data)
            title = soup.title.get_text(" ", strip=True) if soup.title else source["title"]
            if ERROR_PAGE.search(title) or ERROR_PAGE.search(soup.get_text(" ", strip=True)[:500]):
                raise ValueError(f"Server returned an error/challenge page: {title}")
            entry["page_title"] = title
            if source.get("discover"):
                for anchor in soup.find_all("a", href=True):
                    href = urldefrag(urljoin(final_url, anchor["href"]))[0]
                    label = anchor.get_text(" ", strip=True)
                    if (
                        urlsplit(href).scheme == "https"
                        and urlsplit(href).netloc == urlsplit(final_url).netloc
                        and KEYWORDS.search(label + " " + href)
                    ):
                        links.append((href, label or title))
            text = html_text(soup)
            if len(text) < 200:
                raise ValueError("Insufficient HTML text; may require JavaScript")
            pages = [{"page": None, "text": text}]
            suffix = ".html"
            entry["format"] = "html"
        raw_path = output / "raw" / (source["id"] + suffix)
        raw_path.write_bytes(data)
        entry["raw_path"] = raw_path.relative_to(ROOT).as_posix()
        entry["page_count"] = len(pages) if suffix == ".pdf" else None
        entry["characters"] = sum(len(page["text"].strip()) for page in pages)
        entry["status"] = "ok" if entry["characters"] >= 200 else "needs_ocr"
        extracted = {
            "document_id": source["id"], "title": source["title"],
            "url": source["url"], "final_url": final_url,
            "category": source["category"], "retrieved_at": entry["retrieved_at"],
            "sha256": entry["sha256"], "format": entry["format"],
            "validity_status": entry["validity_status"], "pages": pages,
        }
        extracted_path = output / "text" / (source["id"] + ".json")
        dump_json(extracted_path, extracted)
        entry["text_path"] = extracted_path.relative_to(ROOT).as_posix()
        return entry, links
    except (requests.RequestException, ValueError, OSError, PyPdfError) as exc:
        # Each source failure is persisted and printed; other independent sources continue.
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return entry, []


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--snapshot", default=datetime.now(timezone.utc).date().isoformat())
    cli.add_argument("--links-per-source", type=int, default=6)
    args = cli.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.snapshot):
        cli.error("--snapshot must be YYYY-MM-DD")
    output = ROOT / "data" / args.snapshot
    if (output / "manifest.json").exists():
        cli.error("Snapshot already exists; use a different snapshot label to preserve it")
    for name in ("raw", "text"):
        (output / name).mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    origins = sorted({f"{urlsplit(s['url']).scheme}://{urlsplit(s['url']).netloc}" for s in sources})
    with ThreadPoolExecutor(max_workers=5) as pool:
        rules = dict(zip(origins, pool.map(robots_for, origins)))
    entries, children, seen = [], [], {s["url"] for s in sources}

    def run(source):
        parts = urlsplit(source["url"])
        return collect_one(source, output, rules[f"{parts.scheme}://{parts.netloc}"])

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(run, sources))
    for source, (entry, links) in zip(sources, results):
        entries.append(entry)
        print(f"{entry['status']:10} {entry['id']} {entry.get('error', '')}", flush=True)
        # Prefer downloadable reference material to site menus/news.
        links.sort(key=lambda item: (not urlsplit(item[0]).path.lower().endswith(".pdf"), item[0]))
        added = 0
        for url, title in links:
            if url in seen or added >= args.links_per_source:
                continue
            seen.add(url)
            added += 1
            children.append({
                "id": source["id"] + "-" + hashlib.sha256(url.encode()).hexdigest()[:10],
                "title": title, "url": url, "category": source["category"],
                "parent_url": source["url"],
            })
    # A second bounded level follows the links actually published by each source.
    with ThreadPoolExecutor(max_workers=5) as pool:
        for entry, _ in pool.map(run, children):
            entries.append(entry)
            print(f"{entry['status']:10} {entry['id']} {entry.get('error', '')}", flush=True)
    dump_json(output / "manifest.json", entries)
    summary = {
        "snapshot_label": args.snapshot,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "documents": len(entries),
        "statuses": {status: sum(e["status"] == status for e in entries)
                     for status in ("ok", "needs_ocr", "blocked", "failed")},
        "pdfs_saved": sum(e.get("format") == "pdf" and "raw_path" in e for e in entries),
        "html_saved": sum(e.get("format") == "html" and "raw_path" in e for e in entries),
        "legal_validity_verified": False,
    }
    dump_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["statuses"]["ok"]:
        raise SystemExit("No usable documents collected; inspect manifest.json")


if __name__ == "__main__":
    main()

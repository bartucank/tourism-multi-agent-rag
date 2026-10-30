"""Download relevant linked PDFs and create an explicitly selected RAG corpus."""

from collections import Counter
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlsplit

from collect import ROOT, collect_one, dump_json, html_text, parse_html, robots_for

ALLOWED_HOSTS = {
    "teftis.ktb.gov.tr", "www.ktb.gov.tr", "yigm.ktb.gov.tr",
    "ticaret.gov.tr", "www.ticaret.gov.tr", "webdosya.kosgeb.gov.tr",
    "www.yatirimadestek.gov.tr", "www.tursab.org.tr", "www.kgf.com.tr",
}
RELEVANT = re.compile(
    r"turizm|konaklama|seyahat|acenta|acente|2634|1618|7183|"
    r"genelge|çerçeve|cerceve|yönerge|yonerge|desteklenensektor|"
    r"vergi|kefalet|teşvik|tesvik|yatırım|yatirim", re.I,
)
EXCLUDE = re.compile(
    r"kişisel|kisisel|gizlilik|imha|aydınlatma|aydinlatma|"
    r"mülga|mulga|arsiv|arşiv|ihale|personel|telefon|irtibat", re.I,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--extend", action="store_true", help="Reuse saved files; fetch new links only")
    cli.add_argument("--snapshot", default="2026-10-16", help="collect.py ile olusturulan YYYY-MM-DD klasoru")
    args = cli.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.snapshot):
        cli.error("--snapshot must be YYYY-MM-DD")
    global OUT
    SNAPSHOT = ROOT / "data" / args.snapshot
    OUT = ROOT / "data" / f"{args.snapshot}-documents"
    existing = []
    if (OUT / "manifest.json").exists() and not args.extend:
        raise SystemExit("Document snapshot already exists; it will not be overwritten")
    if (OUT / "manifest.json").exists():
        existing = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for folder in ("raw", "text"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    original = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    # Only substantive selected pages enter retrieval; homepages and menus stay in raw.
    selected, candidates, visited = [], [], set()
    per_host = Counter()
    for entry in original:
        if entry["status"] != "ok" or not entry.get("parent_url"):
            continue
        description = entry["title"] + " " + entry["url"]
        keep = (
            entry["id"].startswith("kgf-") and
            any(word in description for word in ("Vadeler", "Süreçleri", "KOSGEB"))
        ) or (
            entry["id"].startswith("kosgeb-") and
            re.search("girisimci|desteklenen-sektor|isletme-gelistirme", entry["url"])
        ) or (
            entry["id"].startswith("ticaret-") and
            re.search("genelge|\\.pdf", entry["url"], re.I)
        ) or (
            entry["id"].startswith("tursab-") and "hizmet-bedeli-tarifesi" in entry["url"]
        )
        if keep and not EXCLUDE.search(description):
            # Re-extract HTML from original bytes with corrected Turkish encoding.
            if entry["format"] == "html":
                document = json.loads((ROOT / entry["text_path"]).read_text(encoding="utf-8"))
                soup = parse_html((ROOT / entry["raw_path"]).read_bytes())
                if soup.title:
                    document["title"] = soup.title.get_text(" ", strip=True)
                document["pages"] = [{"page": None, "text": html_text(soup)}]
                target = OUT / "text" / (entry["id"] + ".json")
                dump_json(target, document)
                entry = dict(entry, title=document["title"],
                             text_path=target.relative_to(ROOT).as_posix())
            selected.append(entry)
        if entry["format"] != "html" or EXCLUDE.search(description):
            continue
        soup = parse_html((ROOT / entry["raw_path"]).read_bytes())
        for anchor in soup.find_all("a", href=True):
            url = urljoin(entry["final_url"], anchor["href"])
            parts = urlsplit(url)
            title = anchor.get_text(" ", strip=True)
            description = title + " " + unquote(parts.path)
            if (
                parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS
                or not parts.path.lower().endswith(".pdf") or url in visited
                or not RELEVANT.search(description) or EXCLUDE.search(description)
                or per_host[parts.hostname] >= 12
            ):
                continue
            visited.add(url)
            per_host[parts.hostname] += 1
            candidates.append({
                "id": "pdf-" + hashlib.sha256(url.encode()).hexdigest()[:12],
                "title": title or Path(unquote(parts.path)).stem,
                "category": entry["category"], "url": url,
                "parent_url": entry["final_url"],
            })
    # The portal publishes tourism-specific summaries below a large general catalogue.
    for entry in original:
        if entry["status"] != "ok" or not entry["id"].startswith("yatirim"):
            continue
        soup = parse_html((ROOT / entry["raw_path"]).read_bytes())
        for anchor in soup.find_all("a", href=True):
            url = urljoin(entry["final_url"], anchor["href"])
            path = unquote(urlsplit(url).path)
            if (
                urlsplit(url).hostname != "www.yatirimadestek.gov.tr"
                or not path.lower().endswith(".pdf") or url in visited
                or not re.search(
                    r"ozet_turizm_kredisi|ozet-turizm_tanitma|"
                    r"ozet-2634|/turizm_tesvikleri\.pdf", path, re.I,
                )
            ):
                continue
            visited.add(url)
            candidates.append({
                "id": "pdf-" + hashlib.sha256(url.encode()).hexdigest()[:12],
                "title": Path(path).stem, "url": url,
                "category": "turizm_destekleri", "parent_url": entry["url"],
            })
    # Follow law links actually present on the ministry's law directory.
    for entry in original:
        if entry["status"] != "ok" or "kanunlar.html" not in entry["url"]:
            continue
        soup = parse_html((ROOT / entry["raw_path"]).read_bytes())
        for anchor in soup.find_all("a", href=True):
            title = anchor.get_text(" ", strip=True)
            url = urljoin(entry["final_url"], anchor["href"])
            if (
                re.search(r"2634|1618|7183|turizm|seyahat", title, re.I)
                and urlsplit(url).hostname in ALLOWED_HOSTS and url not in visited
                and not EXCLUDE.search(title + " " + url)
                and re.search(r"\b(?:2634|1618|7183|7464)\b", title)
            ):
                visited.add(url)
                candidates.append({
                    "id": "law-" + hashlib.sha256(url.encode()).hexdigest()[:12],
                    "title": title, "category": "turizm_mevzuati",
                    "url": url, "parent_url": entry["url"],
                })
    saved_urls = {e["url"] for e in existing}
    candidates = [s for s in candidates if s["url"] not in saved_urls]
    origins = {f"https://{urlsplit(s['url']).netloc}" for s in candidates}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rules = dict(zip(sorted(origins), pool.map(robots_for, sorted(origins))))

    def run(source):
        return collect_one(source, OUT, rules[f"https://{urlsplit(source['url']).netloc}"])[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = existing + list(pool.map(run, candidates))
    for entry in entries:
        print(entry["status"], entry["title"][:80], entry.get("error", ""), flush=True)
    dump_json(OUT / "manifest.json", entries)
    corpus = selected + [e for e in entries if e["status"] == "ok"]
    # Drop byte-identical documents, retaining their first source attribution.
    corpus = list({e["sha256"]: e for e in reversed(corpus)}.values())
    corpus.reverse()
    excluded = []
    curated = []
    for entry in corpus:
        url = unquote(entry["url"]).lower()
        description = entry["title"] + " " + url
        # Raw downloads are an archive, not a guarantee of retrieval relevance.
        keep = (
            entry["id"].startswith("kgf-")
            or entry["id"].startswith("kosgeb-")
            or "hizmet-sektorlerinin-desteklenmesine" in url
            or "çerçeve karar.pdf" in url
            or "dys_genelgesi" in url
            or "desteklenensektorler.pdf" in url
            or "gdp_yo" in url
            or "3065-sayili" in url
            or entry["category"] == "turizm_destekleri"
            or "2025-yilsonu-degerlendirmesi.pdf" in url
            or "c871637b-4723-4944-900d-a894c11fb787.pdf" in url
            or "hizmet-bedeli-tarifesi" in url
            or re.search(r"\b2634\b", entry["title"]) is not None
        )
        if keep and not EXCLUDE.search(description):
            if entry["title"] in {"İNDİR", "DETAYLI BİLGİ"}:
                entry["title"] = Path(urlsplit(url).path).stem
            curated.append(entry)
        else:
            excluded.append({"id": entry["id"], "title": entry["title"],
                             "reason": "Historical, repealed or outside demo scope"})
    corpus = curated
    dump_json(ROOT / "data" / "excluded.json", excluded)
    dump_json(ROOT / "data" / "corpus.json", corpus)
    print(f"Selected corpus: {len(corpus)} documents; PDFs: "
          f"{sum(e['format'] == 'pdf' for e in corpus)}")
    if not corpus:
        raise SystemExit("No documents available for RAG")


if __name__ == "__main__":
    main()

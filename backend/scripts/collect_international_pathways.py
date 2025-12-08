"""
Collect comprehensive bachelor-level program data for selected countries and
write a normalized JSON to backend/data/academic/international_pathways.json.

Important:
- This script requires network access to fetch official university pages.
- It uses heuristics to discover and parse program pages, which may need
  per-university tuning for perfect coverage.

Usage examples:
  python backend/scripts/collect_international_pathways.py \
    --countries "United Kingdom,United States,Australia,Canada,Germany,Netherlands,Ireland,Singapore,New Zealand,Malaysia" \
    --seeds backend/data/academic/seeds.sample.json \
    --output backend/data/academic/international_pathways.json

Then generate PDFs:
  python backend/scripts/generate_international_pathways_pdf.py --per-country

Dependencies: requests, beautifulsoup4 (already in requirements)
Optional: duckduckgo_search/ddgs for discovery (in requirements)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


DEFAULT_COUNTRIES = [
    "United Kingdom",
    "United States",
    "Australia",
    "Canada",
    "Germany",
    "Netherlands",
    "Ireland",
    "Singapore",
    "New Zealand",
    "Malaysia",
]


@dataclass
class Scholarship:
    name: str
    url: str


@dataclass
class FXRate:
    src: str
    dst: str
    rate: float
    as_of: str


@dataclass
class LocalCost:
    amount: float
    currency: str


@dataclass
class Program:
    name: str
    duration: Optional[str] = None
    cost_usd_per_year: Optional[float] = None
    cost_usd_total: Optional[float] = None
    cost_local: Optional[LocalCost] = None
    scholarships: Optional[List[Scholarship]] = None
    url: Optional[str] = None
    note: Optional[str] = None
    as_of: Optional[str] = None
    fx_rate: Optional[FXRate] = None


@dataclass
class University:
    name: str
    website: str
    programs: List[Program]


@dataclass
class CountryData:
    name: str
    universities: List[University]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PathwaysCollector/1.0; +https://example.local)",
}


def fetch(url: str, timeout: int = 20) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            return resp.text
    except Exception:
        return None
    return None


def abs_url(base: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    # Simple join; for complex cases, urljoin (but keep deps minimal)
    if href.startswith("/"):
        # get scheme+host from base
        m = re.match(r"^(https?://[^/]+)/?", base)
        if m:
            return m.group(1) + href
    # relative
    if base.endswith("/"):
        return base + href
    else:
        return base.rsplit("/", 1)[0] + "/" + href


DISCOVERY_HINTS = [
    "undergraduate",
    "bachelor",
    "bachelors",
    "programs",
    "courses",
    "study",
]

PROGRAM_NAME_HINTS = [
    "Bachelor",
    "BSc",
    "BA ",
    "BEng",
    "BComp",
    "LLB",
    "BCom",
    "BBA",
    "BMed",
    "MBBS",
]

FEE_HINTS = [
    "tuition",
    "fee",
    "fees",
    "international",
]

DURATION_HINTS = [
    "duration",
    "years",
]

SCHOLARSHIP_HINTS = [
    "scholarship",
    "scholarships",
    "bursary",
    "financial aid",
]


def discover_program_list_pages(base_url: str) -> List[str]:
    html = fetch(base_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").lower()
        href = a["href"].lower()
        if any(h in text for h in DISCOVERY_HINTS) or any(h in href for h in DISCOVERY_HINTS):
            try:
                links.append(abs_url(base_url, a["href"]))
            except Exception:
                pass
    # De-duplicate
    return list(dict.fromkeys(links))[:50]


def extract_number_and_currency(text: str) -> Optional[Tuple[float, str]]:
    # Simple extraction: find patterns like $12,345 or 12,345 AUD or EUR 12,345
    # Returns (amount, currency_code_or_symbol)
    cur_patterns = [
        (r"USD\s*\$?([0-9][0-9,\.]+)", "USD"),
        (r"\$\s*([0-9][0-9,\.]+)", "$"),
        (r"GBP\s*£?([0-9][0-9,\.]+)", "GBP"),
        (r"£\s*([0-9][0-9,\.]+)", "GBP"),
        (r"EUR\s*€?([0-9][0-9,\.]+)", "EUR"),
        (r"€\s*([0-9][0-9,\.]+)", "EUR"),
        (r"AUD\s*\$?([0-9][0-9,\.]+)", "AUD"),
        (r"SGD\s*\$?([0-9][0-9,\.]+)", "SGD"),
        (r"CAD\s*\$?([0-9][0-9,\.]+)", "CAD"),
        (r"NZD\s*\$?([0-9][0-9,\.]+)", "NZD"),
        (r"MYR\s*([0-9][0-9,\.]+)", "MYR"),
    ]
    for pat, code in cur_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            amt_s = m.group(1).replace(",", "")
            try:
                return float(amt_s), code
            except Exception:
                continue
    return None


def convert_to_usd(amount: float, currency: str, fx_map: Dict[str, float]) -> Optional[Tuple[float, FXRate]]:
    currency = currency.upper()
    if currency == "USD" or currency == "$":
        return amount, FXRate(src="USD", dst="USD", rate=1.0, as_of=datetime.utcnow().date().isoformat())
    rate = fx_map.get(currency)
    if rate:
        return amount * rate, FXRate(src=currency, dst="USD", rate=rate, as_of=datetime.utcnow().date().isoformat())
    return None


def load_fx_rates(fx_path: Optional[Path]) -> Dict[str, float]:
    if fx_path and fx_path.exists():
        try:
            data = json.loads(fx_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k.upper(): float(v) for k, v in data.items()}
        except Exception:
            pass
    # Fallback static approximations (periodically update manually):
    return {
        "GBP": 1.29,
        "EUR": 1.09,
        "AUD": 0.67,
        "SGD": 0.74,
        "CAD": 0.73,
        "NZD": 0.61,
        "MYR": 0.21,
    }


def parse_program_page(url: str, html: str, fx_map: Dict[str, float]) -> Program:
    soup = BeautifulSoup(html, "html.parser")
    # Program name
    name = soup.find("h1")
    prog_name = (name.get_text(strip=True) if name else None) or (soup.title.get_text(strip=True) if soup.title else url)
    # Duration
    duration = None
    text = soup.get_text(" \n ")
    # Look for 'Duration: X years' or similar
    dur_m = re.search(r"Duration\s*:?[ \t]*([A-Za-z0-9 \-–+\(\)]+)", text, re.IGNORECASE)
    if dur_m:
        duration = dur_m.group(1).strip()
    else:
        yrs_m = re.search(r"([2-7])\s*(year|years)\b", text, re.IGNORECASE)
        if yrs_m:
            duration = f"{yrs_m.group(1)} years"
    # Fees (prefer international)
    usd_per_year: Optional[float] = None
    usd_total: Optional[float] = None
    local_cost: Optional[LocalCost] = None
    fx_meta: Optional[FXRate] = None
    # Scan text blocks around 'tuition'/'fee'
    for block in re.split(r"\n{2,}", text):
        if any(h in block.lower() for h in FEE_HINTS):
            ex = extract_number_and_currency(block)
            if ex:
                amt, cur = ex
                local_cost = LocalCost(amount=amt, currency=cur)
                conv = convert_to_usd(amt, cur, fx_map)
                if conv:
                    usd, fx = conv
                    # Heuristic: assume per year if block contains 'per year'
                    if re.search(r"per\s*year|annum|annual", block, re.IGNORECASE):
                        usd_per_year, fx_meta = usd, fx
                    else:
                        # If 'per year' is not explicit, store as total to avoid overstating
                        usd_total, fx_meta = usd, fx
                    break
    return Program(
        name=prog_name,
        duration=duration,
        cost_usd_per_year=usd_per_year,
        cost_usd_total=usd_total,
        cost_local=local_cost,
        scholarships=None,
        url=url,
        note=None,
        as_of=datetime.utcnow().date().isoformat(),
        fx_rate=fx_meta,
    )


def extract_program_links(page_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[str] = []
    for a in soup.find_all("a", href=True):
        label = a.get_text() or ""
        href = a["href"]
        if any(h in label for h in PROGRAM_NAME_HINTS) or any(h in href for h in ["/bachelor", "/undergraduate", "/program", "/course"]):
            try:
                out.append(abs_url(page_url, href))
            except Exception:
                pass
    return list(dict.fromkeys(out))[:200]


def find_scholarships(base_url: str) -> List[Scholarship]:
    html = fetch(base_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[Tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip()
        href = a["href"]
        t_low = text.lower()
        h_low = href.lower()
        if (any(h in t_low for h in SCHOLARSHIP_HINTS) or any(h in h_low for h in SCHOLARSHIP_HINTS)) and len(text) > 3:
            try:
                candidates.append((text, abs_url(base_url, href)))
            except Exception:
                pass
    # Deduplicate by URL
    uniq: Dict[str, Scholarship] = {}
    for name, url in candidates:
        if url not in uniq:
            uniq[url] = Scholarship(name=name, url=url)
    # Return a few most relevant
    return list(uniq.values())[:6]


def collect_for_university(name: str, website: str, fx_map: Dict[str, float], per_uni_limit: Optional[int] = None) -> University:
    # Discover entry points for programs
    list_pages = discover_program_list_pages(website)
    program_urls: List[str] = []
    for lp in list_pages:
        time.sleep(0.6)
        html = fetch(lp)
        if not html:
            continue
        program_urls.extend(extract_program_links(lp, html))
        if per_uni_limit and len(program_urls) >= per_uni_limit:
            break
    # De-duplicate
    program_urls = list(dict.fromkeys(program_urls))
    programs: List[Program] = []
    for pu in program_urls[: (per_uni_limit or len(program_urls))]:
        time.sleep(0.6)
        html = fetch(pu)
        if not html:
            continue
        try:
            p = parse_program_page(pu, html, fx_map)
            programs.append(p)
        except Exception:
            continue
    # Scholarships
    scholarships = find_scholarships(website)
    if scholarships:
        # Attach to all programs as general options
        for p in programs:
            p.scholarships = (p.scholarships or []) + scholarships
    return University(name=name, website=website, programs=programs)


def load_seeds(seeds_path: Optional[Path]) -> Dict[str, List[Dict[str, str]]]:
    if seeds_path and seeds_path.exists():
        try:
            data = json.loads(seeds_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect international pathways data (bachelor-level, all disciplines)")
    ap.add_argument("--countries", default=",".join(DEFAULT_COUNTRIES), help="Comma-separated list of countries")
    ap.add_argument("--seeds", default=None, help="Path to JSON with initial university seeds per country")
    ap.add_argument("--output", default=None, help="Output JSON path; defaults to backend/data/academic/international_pathways.json")
    ap.add_argument("--fx", default=None, help="Path to JSON with FX rates to USD { 'EUR': 1.09, ... }")
    ap.add_argument("--per-university-limit", type=int, default=None, help="Limit programs per university during initial collection (for testing)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "backend" / "data" / "academic"
    out_path = Path(args.output) if args.output else (data_dir / "international_pathways.json")

    ensure_dir = out_path.parent
    ensure_dir.mkdir(parents=True, exist_ok=True)

    fx_map = load_fx_rates(Path(args.fx) if args.fx else None)
    seeds = load_seeds(Path(args.seeds) if args.seeds else None)
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    results: List[CountryData] = []
    for country in countries:
        seed_unis = seeds.get(country, [])
        country_unis: List[University] = []
        print(f"Collecting {country} ... seeds: {len(seed_unis)}")
        for su in seed_unis:
            name = su.get("name")
            site = su.get("website")
            if not name or not site:
                continue
            try:
                u = collect_for_university(name, site, fx_map, per_uni_limit=args.per_university_limit)
                # Only include universities where at least one program was found
                if u.programs:
                    country_unis.append(u)
                    print(f"  + {name}: {len(u.programs)} programs")
                else:
                    print(f"  - {name}: no programs discovered (heuristics may need tuning)")
            except Exception as e:
                print(f"  ! {name}: error {e}")
        results.append(CountryData(name=country, universities=country_unis))

    # Serialize to JSON
    def to_jsonable(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            d = asdict(obj)
            # Flatten dataclasses inside lists as well
            return d
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        return obj

    output = [to_jsonable(c) for c in results]
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


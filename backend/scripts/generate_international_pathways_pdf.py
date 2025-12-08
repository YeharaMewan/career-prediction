"""
Generate International Pathways PDF into backend/data/academic.

Outputs by default: backend/data/academic/International_Pathways.pdf
Optionally: per-country PDFs when using --per-country

Reads data from backend/data/academic/international_pathways.json if present.
Falls back to built-in example data.

Requires: weasyprint (preferred, but optional) or PyMuPDF fallback.
"""
from __future__ import annotations

import os
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional

# Try WeasyPrint first; if not available (or missing system libs),
# we will fallback to PyMuPDF (pymupdf) for PDF generation.
try:
    from weasyprint import HTML  # type: ignore
    _WEASY_AVAILABLE = True
except Exception:  # ImportError or OSError due to missing native deps
    _WEASY_AVAILABLE = False
    HTML = None  # type: ignore

try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
except Exception:
    _PYMUPDF_AVAILABLE = False


def build_data() -> List[Dict[str, Any]]:
    """Return structured data for International Pathways.

    Countries exclude Sri Lanka, and include for each university:
    - program name, duration, cost, scholarships, website URL, and a short note.

    Data based on txt/acadamicoutput.txt example content with added official URLs.
    """
    return [
        {
            "name": "United Kingdom",
            "universities": [
                {
                    "name": "University of London",
                    "website": "https://www.london.ac.uk",
                    "programs": [
                        {
                            "name": "BSc Computer Science",
                            "duration": "3 years (full-time)",
                            "cost": "$22,000-35,000 per year (tuition + living)",
                            "scholarships": [
                                "Chevening Scholarship",
                                "Commonwealth Scholarships",
                                "University merit scholarships",
                            ],
                            "url": "https://www.london.ac.uk/study/undergraduate",
                            "note": "Shorter duration than US; 2-year post-study work visa.",
                        }
                    ],
                },
                {
                    "name": "Manchester Metropolitan University",
                    "website": "https://www.mmu.ac.uk",
                    "programs": [
                        {
                            "name": "BSc (Hons) Computer Science",
                            "duration": "3 years (full-time)",
                            "cost": "$22,000-35,000 per year (tuition + living)",
                            "scholarships": [
                                "Chevening Scholarship",
                                "Commonwealth Scholarships",
                                "University merit scholarships",
                            ],
                            "url": "https://www.mmu.ac.uk/study/undergraduate",
                            "note": "Shorter duration than US; 2-year post-study work visa.",
                        }
                    ],
                },
                {
                    "name": "Coventry University",
                    "website": "https://www.coventry.ac.uk",
                    "programs": [
                        {
                            "name": "BSc (Hons) Computer Science",
                            "duration": "3 years (full-time)",
                            "cost": "$22,000-35,000 per year (tuition + living)",
                            "scholarships": [
                                "Chevening Scholarship",
                                "Commonwealth Scholarships",
                                "University merit scholarships",
                            ],
                            "url": "https://www.coventry.ac.uk/study-at-coventry",
                            "note": "Shorter duration than US; 2-year post-study work visa.",
                        }
                    ],
                },
                {
                    "name": "University of Westminster",
                    "website": "https://www.westminster.ac.uk",
                    "programs": [
                        {
                            "name": "BSc (Hons) Computer Science",
                            "duration": "3 years (full-time)",
                            "cost": "$22,000-35,000 per year (tuition + living)",
                            "scholarships": [
                                "Chevening Scholarship",
                                "Commonwealth Scholarships",
                                "University merit scholarships",
                            ],
                            "url": "https://www.westminster.ac.uk/courses/undergraduate",
                            "note": "Shorter duration than US; 2-year post-study work visa.",
                        }
                    ],
                },
                {
                    "name": "University of Plymouth",
                    "website": "https://www.plymouth.ac.uk",
                    "programs": [
                        {
                            "name": "BSc (Hons) Computing",
                            "duration": "3 years (full-time)",
                            "cost": "$22,000-35,000 per year (tuition + living)",
                            "scholarships": [
                                "Chevening Scholarship",
                                "Commonwealth Scholarships",
                                "University merit scholarships",
                            ],
                            "url": "https://www.plymouth.ac.uk/courses",
                            "note": "Shorter duration than US; 2-year post-study work visa.",
                        }
                    ],
                },
            ],
        },
        {
            "name": "United States",
            "universities": [
                {
                    "name": "Arizona State University",
                    "website": "https://www.asu.edu",
                    "programs": [
                        {
                            "name": "BS Computer Science",
                            "duration": "4 years (full-time)",
                            "cost": "$25,000-50,000 per year (varies widely)",
                            "scholarships": [
                                "University scholarships (merit/need-based)",
                                "Fulbright Program",
                                "Athletic scholarships",
                            ],
                            "url": "https://degrees.asu.edu",
                            "note": "Many universities offer financial aid to international students.",
                        }
                    ],
                },
                {
                    "name": "Penn State University",
                    "website": "https://www.psu.edu",
                    "programs": [
                        {
                            "name": "BS Computer Science",
                            "duration": "4 years (full-time)",
                            "cost": "$25,000-50,000 per year (varies widely)",
                            "scholarships": [
                                "University scholarships (merit/need-based)",
                                "Fulbright Program",
                                "Athletic scholarships",
                            ],
                            "url": "https://bulletins.psu.edu/undergraduate/colleges/",
                            "note": "Many universities offer financial aid to international students.",
                        }
                    ],
                },
                {
                    "name": "University of Illinois (Urbana-Champaign)",
                    "website": "https://illinois.edu",
                    "programs": [
                        {
                            "name": "BS Computer Science",
                            "duration": "4 years (full-time)",
                            "cost": "$25,000-50,000 per year (varies widely)",
                            "scholarships": [
                                "University scholarships (merit/need-based)",
                                "Fulbright Program",
                                "Athletic scholarships",
                            ],
                            "url": "https://cs.illinois.edu/academics/undergraduate",
                            "note": "Many universities offer financial aid to international students.",
                        }
                    ],
                },
                {
                    "name": "Purdue University",
                    "website": "https://www.purdue.edu",
                    "programs": [
                        {
                            "name": "BS Computer Science",
                            "duration": "4 years (full-time)",
                            "cost": "$25,000-50,000 per year (varies widely)",
                            "scholarships": [
                                "University scholarships (merit/need-based)",
                                "Fulbright Program",
                                "Athletic scholarships",
                            ],
                            "url": "https://www.cs.purdue.edu/academic-programs/undergraduate/",
                            "note": "Many universities offer financial aid to international students.",
                        }
                    ],
                },
                {
                    "name": "Community Colleges with transfer pathways",
                    "website": "https://www.aacc.nche.edu",
                    "programs": [
                        {
                            "name": "Associate in Computer Science (transfer pathway)",
                            "duration": "2 + 2 years (typical)",
                            "cost": "$10,000-30,000 per year (varies widely)",
                            "scholarships": [
                                "Institutional scholarships (varies by college)",
                                "State/Local opportunities",
                                "Foundation scholarships",
                            ],
                            "url": "https://www.aacc.nche.edu/learn-college-costs/",
                            "note": "Cost-effective start; then transfer to a 4-year university.",
                        }
                    ],
                },
            ],
        },
        {
            "name": "Australia",
            "universities": [
                {
                    "name": "Monash University",
                    "website": "https://www.monash.edu",
                    "programs": [
                        {
                            "name": "Bachelor of Computer Science",
                            "duration": "3-4 years",
                            "cost": "$20,000-35,000 per year (AUD)",
                            "scholarships": [
                                "Australia Awards",
                                "University scholarships",
                                "Destination Australia",
                            ],
                            "url": "https://www.monash.edu/study/courses",
                            "note": "Post-study work rights available; pathway to permanent residence.",
                        }
                    ],
                },
                {
                    "name": "University of Melbourne",
                    "website": "https://www.unimelb.edu.au",
                    "programs": [
                        {
                            "name": "Bachelor of Science (Computing & Software Systems)",
                            "duration": "3-4 years",
                            "cost": "$20,000-35,000 per year (AUD)",
                            "scholarships": [
                                "Australia Awards",
                                "University scholarships",
                                "Destination Australia",
                            ],
                            "url": "https://study.unimelb.edu.au/find/",
                            "note": "Post-study work rights available; pathway to permanent residence.",
                        }
                    ],
                },
                {
                    "name": "Curtin University",
                    "website": "https://www.curtin.edu.au",
                    "programs": [
                        {
                            "name": "Bachelor of Computing",
                            "duration": "3-4 years",
                            "cost": "$20,000-35,000 per year (AUD)",
                            "scholarships": [
                                "Australia Awards",
                                "University scholarships",
                                "Destination Australia",
                            ],
                            "url": "https://www.curtin.edu.au/study/find-a-course/",
                            "note": "Post-study work rights available; pathway to permanent residence.",
                        }
                    ],
                },
                {
                    "name": "Victoria University",
                    "website": "https://www.vu.edu.au",
                    "programs": [
                        {
                            "name": "Bachelor of Information Technology",
                            "duration": "3-4 years",
                            "cost": "$20,000-35,000 per year (AUD)",
                            "scholarships": [
                                "Australia Awards",
                                "University scholarships",
                                "Destination Australia",
                            ],
                            "url": "https://www.vu.edu.au/courses",
                            "note": "Post-study work rights available; pathway to permanent residence.",
                        }
                    ],
                },
                {
                    "name": "RMIT University",
                    "website": "https://www.rmit.edu.au",
                    "programs": [
                        {
                            "name": "Bachelor of Computer Science",
                            "duration": "3-4 years",
                            "cost": "$20,000-35,000 per year (AUD)",
                            "scholarships": [
                                "Australia Awards",
                                "University scholarships",
                                "Destination Australia",
                            ],
                            "url": "https://www.rmit.edu.au/study",
                            "note": "Post-study work rights available; pathway to permanent residence.",
                        }
                    ],
                },
            ],
        },
    ]


def format_usd(n: Optional[float]) -> Optional[str]:
    if n is None:
        return None
    try:
        return f"${int(round(float(n))):,}"
    except Exception:
        return None


def render_html(data: List[Dict[str, Any]]) -> str:
    """Render a simple, clean HTML document from the data."""
    def esc(txt: str) -> str:
        return (txt or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts: List[str] = []
    parts.append(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>International Pathways</title>
    <style>
      @page { size: A4; margin: 24mm 18mm; }
      body { font-family: Arial, Helvetica, sans-serif; color: #1b1b1b; }
      h1.title { font-size: 28px; margin: 0 0 4px 0; }
      p.subtitle { color: #444; margin: 0 0 24px 0; }
      h2.country { font-size: 22px; margin: 28px 0 12px; color: #0a4cbf; }
      h3.university { font-size: 16px; margin: 14px 0 6px; }
      .block { border: 1px solid #dcdcdc; border-radius: 8px; padding: 10px 12px; margin: 8px 0 14px; }
      .label { color: #222; font-weight: bold; }
      .value { color: #333; }
      ul { margin: 6px 0 6px 20px; }
      a { color: #0a66c2; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .note { color: #4a4a4a; font-style: italic; margin-top: 4px; }
      .meta { color: #666; font-size: 12px; }
      .country-break { page-break-after: always; }
      .country:last-of-type .country-break { page-break-after: auto; }
    </style>
  </head>
  <body>
    <h1 class="title">International Pathways</h1>
    <p class="subtitle">Countries and universities outside Sri Lanka with programs, durations, costs, scholarships, and website links.</p>
"""
    )

    for ci, country in enumerate(data):
        parts.append(f'<div class="country">')
        parts.append(f'<h2 class="country">{esc(country["name"])}</h2>')
        for ui, uni in enumerate(country.get("universities", [])):
            parts.append('<div class="block">')
            parts.append(f'<h3 class="university">{esc(uni["name"])}</h3>')
            website = uni.get("website")
            if website:
                parts.append(f'<div><span class="label">University site:</span> <a href="{esc(website)}">{esc(website)}</a></div>')

            programs = uni.get("programs")
            if programs:
                for prog in programs:
                    parts.append('<div style="margin:8px 0 8px 8px; padding-left:8px; border-left:3px solid #e0e0e0;">')
                    parts.append(f'<div><span class="label">Program:</span> <span class="value">{esc(prog.get("name", ""))}</span></div>')
                    if prog.get("duration"):
                        parts.append(f'<div><span class="label">Duration:</span> <span class="value">{esc(prog["duration"])}</span></div>')
                    usd_year = prog.get("cost_usd_per_year")
                    usd_total = prog.get("cost_usd_total")
                    cost_str = None
                    if usd_year is not None:
                        fs = format_usd(usd_year)
                        if fs:
                            cost_str = f"{fs} per year"
                    elif usd_total is not None:
                        fs = format_usd(usd_total)
                        if fs:
                            cost_str = f"{fs} total"
                    if cost_str:
                        parts.append(f'<div><span class="label">Cost (USD):</span> <span class="value">{cost_str}</span></div>')
                    # Optional local currency transparency
                    if prog.get("cost_local"):
                        try:
                            amt = prog["cost_local"].get("amount")
                            cur = prog["cost_local"].get("currency")
                            if amt and cur:
                                parts.append(f'<div class="meta">Local cost: {esc(str(amt))} {esc(cur)}</div>')
                        except Exception:
                            pass
                    scholarships = prog.get("scholarships", [])
                    if scholarships:
                        parts.append('<div><span class="label">Scholarships:</span>')
                        parts.append('<ul>')
                        for s in scholarships:
                            if isinstance(s, dict):
                                nm = esc(str(s.get("name", "")))
                                url = esc(str(s.get("url", "")))
                                if url and nm:
                                    parts.append(f'<li class="value"><a href="{url}">{nm}</a></li>')
                                elif nm:
                                    parts.append(f'<li class="value">{nm}</li>')
                            else:
                                parts.append(f'<li class="value">{esc(str(s))}</li>')
                        parts.append('</ul></div>')
                    if prog.get("url"):
                        parts.append(f'<div><span class="label">Program URL:</span> <a href="{esc(prog["url"]) }">{esc(prog["url"])}</a></div>')
                    if prog.get("note"):
                        parts.append(f'<div class="note">{esc(prog["note"])}</div>')
                    if prog.get("as_of"):
                        parts.append(f'<div class="meta">As of: {esc(prog["as_of"])}</div>')
                    parts.append('</div>')
            parts.append('</div>')
        # Add a page break between countries, except after the last
        if ci < len(data) - 1:
            parts.append('<div class="country-break"></div>')
        parts.append('</div>')

    parts.append(
        """
  </body>
</html>
"""
    )
    return "\n".join(parts)


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def write_pdf_with_weasyprint(output_pdf: Path, data: List[Dict[str, Any]]) -> None:
    if not _WEASY_AVAILABLE:
        raise RuntimeError("WeasyPrint not available")
    html = render_html(data)
    HTML(string=html, base_url=str(output_pdf.parent)).write_pdf(str(output_pdf))

def build_country_block_text(country: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Country: {country['name']}")
    lines.append("")
    for uni in country.get("universities", []):
        lines.append(f"University: {uni['name']}")
        if uni.get("website"):
            lines.append(f"University site: {uni['website']}")
        progs = uni.get("programs") or []
        for prog in progs:
            lines.append(f"- Program: {prog.get('name','')}")
            if prog.get("duration"):
                lines.append(f"  Duration: {prog['duration']}")
            if prog.get("cost"):
                lines.append(f"  Cost: {prog['cost']}")
            sch = prog.get("scholarships", [])
            if sch:
                lines.append(f"  Scholarships:")
                for s in sch:
                    lines.append(f"    - {s}")
            if prog.get("url"):
                lines.append(f"  Program URL: {prog['url']}")
            if prog.get("note"):
                lines.append(f"  Note: {prog['note']}")
            lines.append("")
        # spacing between countries
        lines.append("")
    return "\n".join(lines)

def load_json_if_exists(json_path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        if json_path.exists():
            import json
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return None

def write_pdf_with_pymupdf(output_pdf: Path, data: List[Dict[str, Any]]) -> None:
    if not _PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF not available")
    doc = fitz.open()
    page_rect = fitz.paper_rect("a4")
    margin_l = 50
    margin_t = 60
    margin_r = 50
    margin_b = 60
    usable_width = int(page_rect.width - margin_l - margin_r)
    line_height = 14  # points

    def new_page(with_title: bool = False):
        page = doc.new_page(width=page_rect.width, height=page_rect.height)
        y = margin_t
        if with_title:
            page.insert_text(
                fitz.Point(margin_l, y),
                "International Pathways",
                fontsize=20,
                fontname="helv",
                fill=(0, 0, 0),
            )
            y += 30
        return page, y

    def wrap_line(text: str, max_chars: int = 95) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        lines: List[str] = []
        current: List[str] = []
        for w in words:
            candidate = (" ".join(current + [w])).strip()
            if len(candidate) <= max_chars:
                current.append(w)
            else:
                lines.append(" ".join(current))
                current = [w]
        if current:
            lines.append(" ".join(current))
        return lines

    first = True
    page, y = new_page(with_title=first)
    first = False
    max_y = page_rect.height - margin_b

    for country in data:
        block_text = build_country_block_text(country)
        for raw_line in block_text.splitlines():
            wrapped = wrap_line(raw_line, max_chars=95)
            for line in wrapped:
                if y + line_height > max_y:
                    page, y = new_page(with_title=False)
                page.insert_text(
                    fitz.Point(margin_l, y),
                    line,
                    fontsize=11,
                    fontname="helv",
                    fill=(0, 0, 0),
                )
                y += line_height
        # extra spacing between countries
        y += line_height

    doc.save(str(output_pdf))
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate International Pathways PDF(s)")
    parser.add_argument("--input", default=None, help="Path to JSON data; defaults to backend/data/academic/international_pathways.json")
    parser.add_argument("--output", default=None, help="Output PDF path; defaults to International_Pathways.pdf in academic folder")
    parser.add_argument("--per-country", action="store_true", help="Also create per-country PDFs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "backend" / "data" / "academic"
    input_json = Path(args.input) if args.input else (data_dir / "international_pathways.json")
    output_pdf = Path(args.output) if args.output else (data_dir / "International_Pathways.pdf")

    data = load_json_if_exists(input_json) or build_data()
    ensure_output_dir(output_pdf)

    def render_to(path: Path, subset: Optional[List[Dict[str, Any]]] = None):
        d = subset if subset is not None else data
        try:
            if _WEASY_AVAILABLE:
                write_pdf_with_weasyprint(path, d)
            else:
                raise RuntimeError("WeasyPrint unavailable")
        except Exception:
            if not _PYMUPDF_AVAILABLE:
                raise
            write_pdf_with_pymupdf(path, d)
        print(f"Generated: {path}")

    # Master PDF
    render_to(output_pdf)

    # Optional per-country PDFs
    if args.per_country:
        for country in data:
            name = country.get("name", "country").strip().replace(" ", "_")
            path = output_pdf.parent / f"International_Pathways_{name}.pdf"
            render_to(path, [country])


if __name__ == "__main__":
    main()

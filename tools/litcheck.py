#!/usr/bin/env python3
"""litcheck: literature novelty check for every project in projects/.

For each project it (1) runs curated queries (tools/litcheck_queries.json) plus an
auto-query built from the README title against several scholarly APIs,
(2) scores every hit by TF-IDF cosine similarity to the project's pitch + gap
text, (3) walks the citation neighbourhood of the top hits (papers citing them,
via Europe PMC) to catch very recent follow-ups, and (4) writes
litcheck/results/<slug>.md and litcheck/SUMMARY.md with a near-miss flag.

Backends (all optional; used when reachable / keyed):
  europepmc  - no key, has abstracts + citing-paper lists (slow, ~10 s/call)
  pubmed     - no key needed (NCBI_API_KEY raises the rate limit)
  arxiv      - no key
  crossref   - no key (set CROSSREF_MAILTO for the polite pool)
  openalex   - needs OPENALEX_API_KEY (free)
  s2         - needs S2_API_KEY (Semantic Scholar)

Usage:
  python tools/litcheck.py                      # all projects
  python tools/litcheck.py brain-age-transportability sepsis-definition-multiverse
  python tools/litcheck.py --backends europepmc,pubmed --since 2023 --no-cite-walk

All raw API responses are cached in litcheck/cache/ (git-ignored), so re-runs are free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
OUT = ROOT / "litcheck"
CACHE = OUT / "cache"
RESULTS = OUT / "results"
QUERIES_FILE = ROOT / "tools" / "litcheck_queries.json"

UA = "litcheck/1.0 (biomedical project novelty check; mailto:%s)" % os.environ.get("CROSSREF_MAILTO", "none@example.org")


# ----------------------------------------------------------------------------- model
@dataclass
class Paper:
    title: str
    year: int | None
    venue: str = ""
    doi: str = ""
    ids: dict = field(default_factory=dict)
    abstract: str = ""
    cited_by: int | None = None
    source: str = ""
    via: str = ""          # which query / walk produced it
    score: float = 0.0

    def key(self) -> str:
        if self.doi:
            return "doi:" + self.doi.lower()
        t = re.sub(r"[^a-z0-9]+", " ", self.title.lower()).strip()
        return "title:" + t[:120]


# ----------------------------------------------------------------------------- http + cache
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_last_call: dict[str, float] = defaultdict(float)
MIN_INTERVAL = {"europepmc": 1.0, "pubmed": 0.4, "arxiv": 3.1, "crossref": 1.0, "openalex": 0.2, "s2": 1.1}


def _throttle(backend: str) -> None:
    with _locks[backend]:
        wait = MIN_INTERVAL.get(backend, 1.0) - (time.time() - _last_call[backend])
        if wait > 0:
            time.sleep(wait)
        _last_call[backend] = time.time()


def http_get(backend: str, url: str, headers: dict | None = None, timeout: int = 60, retries: int = 3) -> bytes | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(url.encode()).hexdigest()
    cf = CACHE / f"{backend}_{h}.bin"
    if cf.exists():
        return cf.read_bytes()
    for attempt in range(retries):
        _throttle(backend)
        req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            cf.write_bytes(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1) * 2)
                continue
            sys.stderr.write(f"[{backend}] HTTP {e.code} {url[:100]}\n")
            return None
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            sys.stderr.write(f"[{backend}] {type(e).__name__}: {url[:100]}\n")
            return None
    return None


def q(s: str) -> str:
    return urllib.parse.quote(s, safe="")


# ----------------------------------------------------------------------------- backends
def search_europepmc(query: str, n: int = 25, since: int = 2018) -> list[Paper]:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
           + q(f"({query}) AND (FIRST_PDATE:[{since}-01-01 TO 2030-12-31])")
           + f"&format=json&pageSize={n}&resultType=core")
    data = http_get("europepmc", url, timeout=90)
    if not data:
        return []
    out = []
    for r in json.loads(data).get("resultList", {}).get("result", []):
        out.append(Paper(
            title=r.get("title", "").rstrip("."), year=int(r["pubYear"]) if r.get("pubYear") else None,
            venue=r.get("journalTitle", "") or r.get("bookOrReportDetails", {}).get("publisher", ""),
            doi=r.get("doi", ""), ids={"src": r.get("source"), "id": r.get("id"), "pmid": r.get("pmid")},
            abstract=r.get("abstractText", "") or "", cited_by=r.get("citedByCount"), source="europepmc"))
    return out


def citing_europepmc(p: Paper, n: int = 50) -> list[Paper]:
    src, pid = p.ids.get("src"), p.ids.get("id")
    if not (src and pid):
        return []
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{src}/{pid}/citations?format=json&pageSize={n}"
    data = http_get("europepmc", url, timeout=90)
    if not data:
        return []
    out = []
    for r in json.loads(data).get("citationList", {}).get("citation", []):
        out.append(Paper(title=r.get("title", "").rstrip("."), year=int(r["pubYear"]) if r.get("pubYear") else None,
                         venue=r.get("journalAbbreviation", ""), ids={"src": r.get("source"), "id": r.get("id")},
                         cited_by=r.get("citedByCount"), source="europepmc", via=f"cites:{p.title[:40]}"))
    return out


def citing_pubmed(p: Paper, n: int = 50) -> list[Paper]:
    """Papers citing p, via NCBI elink (pubmed_pubmed_citedin); fallback when Europe PMC is down."""
    pmid = p.ids.get("pmid")
    if not pmid:
        return []
    key = os.environ.get("NCBI_API_KEY", "")
    kp = f"&api_key={key}" if key else ""
    data = http_get("pubmed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pubmed"
                    f"&linkname=pubmed_pubmed_citedin&retmode=json&id={pmid}{kp}")
    if not data:
        return []
    ids = []
    for ls in json.loads(data).get("linksets", []):
        for ldb in ls.get("linksetdbs", []):
            ids += [l["id"] for l in ldb.get("links", [])]
    ids = ids[:n]
    if not ids:
        return []
    ps = _pubmed_fetch(ids)
    for x in ps:
        x.via = f"cites:{p.title[:40]}"
    return ps


def _pubmed_fetch(ids: list[str]) -> list[Paper]:
    key = os.environ.get("NCBI_API_KEY", "")
    kp = f"&api_key={key}" if key else ""
    ef = http_get("pubmed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id="
                  + ",".join(ids) + kp)
    if not ef:
        return []
    out = []
    try:
        root = ET.fromstring(ef)
    except ET.ParseError:
        return []
    for art in root.iter("PubmedArticle"):
        title = "".join((art.find(".//ArticleTitle") is not None and art.find(".//ArticleTitle").itertext()) or "")
        year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate", "")[:4]
        abstract = " ".join("".join(a.itertext()) for a in art.findall(".//AbstractText"))
        doi = ""
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""
        pmid = art.findtext(".//PMID", "")
        out.append(Paper(title=title.rstrip("."), year=int(year) if year and year.isdigit() else None,
                         venue=art.findtext(".//Journal/ISOAbbreviation", ""), doi=doi,
                         ids={"pmid": pmid, "src": "MED", "id": pmid}, abstract=abstract, source="pubmed"))
    return out


def search_pubmed(query: str, n: int = 20, since: int = 2018) -> list[Paper]:
    key = os.environ.get("NCBI_API_KEY", "")
    kp = f"&api_key={key}" if key else ""
    es = http_get("pubmed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json"
                  f"&retmax={n}&sort=relevance&mindate={since}&maxdate=2030&datetype=pdat&term={q(query)}{kp}")
    if not es:
        return []
    ids = json.loads(es).get("esearchresult", {}).get("idlist", [])
    return _pubmed_fetch(ids) if ids else []


def search_arxiv(query: str, n: int = 15, since: int = 2018) -> list[Paper]:
    data = http_get("arxiv", "https://export.arxiv.org/api/query?max_results=" + str(n)
                    + "&sortBy=relevance&search_query=" + q(f"all:{query}"))
    if not data:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return []
    for e in root.findall("a:entry", ns):
        year = int((e.findtext("a:published", "", ns) or "0000")[:4]) or None
        if year and year < since:
            continue
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append(Paper(title=" ".join((e.findtext("a:title", "", ns) or "").split()), year=year, venue="arXiv",
                         doi=e.findtext("{http://arxiv.org/schemas/atom}doi", "", ns) or "", ids={"arxiv": aid},
                         abstract=" ".join((e.findtext("a:summary", "", ns) or "").split()), source="arxiv"))
    return out


def search_crossref(query: str, n: int = 15, since: int = 2018) -> list[Paper]:
    mail = os.environ.get("CROSSREF_MAILTO", "")
    data = http_get("crossref", "https://api.crossref.org/works?rows=" + str(n) + "&query.bibliographic=" + q(query)
                    + f"&filter=from-pub-date:{since}&select=DOI,title,abstract,issued,container-title,is-referenced-by-count"
                    + (f"&mailto={q(mail)}" if mail else ""))
    if not data:
        return []
    out = []
    for it in json.loads(data).get("message", {}).get("items", []):
        parts = (it.get("issued") or {}).get("date-parts") or [[None]]
        abstract = re.sub(r"<[^>]+>", " ", it.get("abstract", "") or "")
        out.append(Paper(title=(it.get("title") or [""])[0], year=parts[0][0], venue=(it.get("container-title") or [""])[0],
                         doi=it.get("DOI", ""), abstract=abstract, cited_by=it.get("is-referenced-by-count"), source="crossref"))
    return out


def search_openalex(query: str, n: int = 25, since: int = 2018) -> list[Paper]:
    key = os.environ.get("OPENALEX_API_KEY")
    if not key:
        return []
    data = http_get("openalex", "https://api.openalex.org/works?per-page=" + str(n) + "&search=" + q(query)
                    + f"&filter=from_publication_date:{since}-01-01"
                    + "&select=id,doi,title,publication_year,primary_location,cited_by_count,abstract_inverted_index",
                    headers={"Authorization": f"Bearer {key}"})
    if not data:
        return []
    out = []
    for w in json.loads(data).get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        words = sorted(((pos, tok) for tok, poss in inv.items() for pos in poss))
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append(Paper(title=w.get("title") or "", year=w.get("publication_year"), venue=loc.get("display_name", ""),
                         doi=(w.get("doi") or "").replace("https://doi.org/", ""), ids={"openalex": w.get("id")},
                         abstract=" ".join(t for _, t in words), cited_by=w.get("cited_by_count"), source="openalex"))
    return out


def search_s2(query: str, n: int = 25, since: int = 2018) -> list[Paper]:
    key = os.environ.get("S2_API_KEY")
    if not key:
        return []
    data = http_get("s2", "https://api.semanticscholar.org/graph/v1/paper/search?limit=" + str(n) + "&query=" + q(query)
                    + f"&year={since}-&fields=title,year,venue,abstract,citationCount,externalIds",
                    headers={"x-api-key": key})
    if not data:
        return []
    out = []
    for p in json.loads(data).get("data", []):
        ext = p.get("externalIds") or {}
        out.append(Paper(title=p.get("title") or "", year=p.get("year"), venue=p.get("venue") or "",
                         doi=ext.get("DOI", ""), ids={"s2": p.get("paperId"), "arxiv": ext.get("ArXiv")},
                         abstract=p.get("abstract") or "", cited_by=p.get("citationCount"), source="s2"))
    return out


BACKENDS = {"europepmc": search_europepmc, "pubmed": search_pubmed, "arxiv": search_arxiv,
            "crossref": search_crossref, "openalex": search_openalex, "s2": search_s2}


# ----------------------------------------------------------------------------- scoring
STOP = set("""a an the of and or to in for on with by from as at is are was were be been being this that these those it its
we our their there which who whom whose what when where how than then also into over under between across via using use used
based both each such not no nor but if while whether per within without about after before during vs versus study studies
data dataset datasets method methods model models analysis approach results result paper project projects propose proposed
novel new first large open public status difficulty timeline compute months month level msc phd weeks week none shipped
starter code design table tables section sections readme folder script scripts module modules test tests file files
env var vars credentials download downloads api rest json csv sql pandas numpy python optional required note notes see
e.g i.e et al fig figure step steps milestone milestones risk risks mitigation ethics venue venues target targets""".split())
_IDF: dict[str, float] | None = None


def tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", s.lower()) if t not in STOP and not t.isdigit()]


def grams(s: str) -> list[str]:
    t = tokens(s)
    return t + [a + " " + b for a, b in zip(t, t[1:])]


def background_idf() -> dict[str, float]:
    """IDF of uni/bigrams across all project READMEs (so project-specific terms weigh most)."""
    global _IDF
    if _IDF is None:
        import math
        df: dict[str, int] = defaultdict(int)
        n = 0
        for d in PROJECTS.iterdir():
            f = d / "README.md"
            if f.exists():
                n += 1
                for g in set(grams(f.read_text(errors="ignore"))):
                    df[g] += 1
        _IDF = {g: math.log((n + 1) / (c + 0.5)) for g, c in df.items()}
        _IDF["__n__"] = n
    return _IDF


def key_terms(text: str, k: int = 40) -> dict[str, float]:
    """Top-k README terms by tf * background idf, weights normalised to sum 1."""
    import math
    idf = background_idf()
    tf: dict[str, int] = defaultdict(int)
    for g in grams(text):
        tf[g] += 1
    default = math.log(idf.get("__n__", 100) + 1)
    w = {g: (1 + math.log(c)) * idf.get(g, default) for g, c in tf.items() if c >= 2 or " " not in g}
    top = sorted(w.items(), key=lambda kv: -kv[1])[:k]
    tot = sum(v for _, v in top) or 1.0
    return {g: v / tot for g, v in top}


def coverage_scores(terms: dict[str, float], docs: list[str]) -> list[float]:
    """Weighted fraction of the project's key terms present in each doc (title+abstract)."""
    out = []
    for d in docs:
        g = set(grams(d))
        out.append(sum(w for t, w in terms.items() if t in g))
    return out


# ----------------------------------------------------------------------------- project text
def readme_text(slug: str) -> tuple[str, str, str]:
    """Return (title, pitch, scoring_text) for a project README."""
    md = (PROJECTS / slug / "README.md").read_text(errors="ignore")
    title = next((l.lstrip("# ").strip() for l in md.splitlines() if l.startswith("# ")), slug)
    paras = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip() and not p.strip().startswith("#")]
    pitch = re.sub(r"[`*_\[\]()|#>]", " ", paras[0] if paras else "")
    pitch = " ".join(pitch.split())
    gap = ""
    m = re.search(r"^#+[^\n]*gap[^\n]*\n(.*?)(?=^#+ )", md, re.S | re.M | re.I)
    if m:
        gap = m.group(1)
    hyp = ""
    m = re.search(r"^#+[^\n]*(hypothes|research question)[^\n]*\n(.*?)(?=^#+ )", md, re.S | re.M | re.I)
    if m:
        hyp = m.group(2)
    text = re.sub(r"[`*_\[\]()|#>]", " ", f"{title}. {pitch} {gap} {hyp}")
    return title, pitch, text[:12000]


def auto_queries(slug: str, title: str, pitch: str, terms: dict[str, float]) -> tuple[str, str]:
    """(long free-text query for arxiv/crossref/epmc, short AND-style query for pubmed)."""
    if slug.replace("-", " ") in title.lower().replace("-", " ") and len(title) < len(slug) + 5:
        long_q = " ".join(pitch.split()[:14])
    else:
        long_q = re.sub(r"[:\-–—/]", " ", title.split("—")[0])[:120]
    uni = [t for t in terms if " " not in t][:5]
    return long_q.strip(), " ".join(uni)


# ----------------------------------------------------------------------------- run one project
def check_project(slug: str, queries: list[str], backends: list[str], since: int, cite_walk: bool,
                  near_year: int, near_sim: float) -> dict:
    title, pitch, ptext = readme_text(slug)
    curated = [x for x in queries if x.strip()]
    terms = key_terms(ptext + (" " + " ".join(curated)) * 3)
    long_q, short_q = auto_queries(slug, title, pitch, terms)
    papers: dict[str, Paper] = {}

    def add(ps: list[Paper], via: str):
        for p in ps:
            if not p.title:
                continue
            k = p.key()
            if k in papers:
                old = papers[k]
                if not old.abstract and p.abstract:
                    old.abstract = p.abstract
                if old.cited_by is None:
                    old.cited_by = p.cited_by
                for kk, vv in p.ids.items():
                    old.ids.setdefault(kk, vv)
                continue
            p.via = p.via or via
            papers[k] = p

    def rescore():
        pl = list(papers.values())
        for p, sc in zip(pl, coverage_scores(terms, [f"{p.title}. {p.title}. {p.abstract}" for p in pl])):
            p.score = sc
        pl.sort(key=lambda p: -p.score)
        return pl

    jobs = []
    for b in backends:
        qs = ([short_q] if b == "pubmed" else [long_q]) + curated
        for qq in qs:
            jobs.append((b, qq))
    with ThreadPoolExecutor(max_workers=max(1, len(backends))) as ex:
        futs = {ex.submit(BACKENDS[b], qq, 25 if b in ("europepmc", "openalex", "s2") else 15, since): (b, qq) for b, qq in jobs}
        for f in as_completed(futs):
            b, qq = futs[f]
            try:
                add(f.result(), f"{b}:{qq[:50]}")
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[{slug}] {b} failed: {e}\n")

    plist = rescore()
    if cite_walk:
        for p in list(plist[:4]):
            got = []
            if "europepmc" in backends and p.ids.get("src") and p.ids.get("id"):
                got = citing_europepmc(p)
            if not got and "pubmed" in backends and p.ids.get("pmid"):
                got = citing_pubmed(p)
            add(got, f"cites:{p.title[:40]}")
        plist = rescore()

    near = [p for p in plist if (p.year or 0) >= near_year and p.score >= near_sim]
    top = plist[:25]
    return {"slug": slug, "title": title, "pitch": pitch[:300], "queries": [long_q, short_q] + curated,
            "key_terms": list(terms)[:15], "n_papers": len(plist),
            "n_recent": sum(1 for p in plist if (p.year or 0) >= near_year),
            "near_misses": [asdict(p) for p in near[:10]], "top": [asdict(p) for p in top],
            "max_sim": top[0].score if top else 0.0,
            "max_recent_sim": max([p.score for p in plist if (p.year or 0) >= near_year], default=0.0)}


def write_result(res: dict, near_year: int) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    L = [f"# Literature check: `{res['slug']}`", "", f"**{res['pitch']}**", "",
         "Key terms: " + ", ".join(res["key_terms"]), "",
         f"Queries: " + " · ".join(f"`{x}`" for x in res["queries"]), "",
         f"Papers retrieved: {res['n_papers']} (≥{near_year}: {res['n_recent']}). Max similarity: {res['max_sim']:.2f}; max similarity among ≥{near_year} papers: {res['max_recent_sim']:.2f}.", ""]
    if res["near_misses"]:
        L += [f"## ⚠️ Near-misses (≥{near_year}, high similarity) — read these before starting", ""]
        for p in res["near_misses"]:
            L.append(f"- **{p['score']:.2f}** {p['year']} — {p['title']} — *{p['venue']}*" + (f" — doi:{p['doi']}" if p['doi'] else "") + (f" — cited by {p['cited_by']}" if p.get('cited_by') is not None else ""))
        L.append("")
    L += ["## Top 25 nearest papers", "", "| sim | year | title | venue | cited | id | via |", "|---|---|---|---|---|---|---|"]
    for p in res["top"]:
        ident = p["doi"] or p["ids"].get("arxiv") or p["ids"].get("pmid") or ""
        L.append(f"| {p['score']:.2f} | {p['year'] or ''} | {p['title'][:140]} | {p['venue'][:40]} | {p['cited_by'] if p['cited_by'] is not None else ''} | {ident} | {p['via'][:30]} |")
    (RESULTS / f"{res['slug']}.md").write_text("\n".join(L) + "\n")


def write_summary(results: list[dict], near_year: int, near_sim: float) -> None:
    results.sort(key=lambda r: -r["max_recent_sim"])
    L = ["# litcheck summary", "", f"Near-miss = paper from ≥{near_year} whose title+abstract covers ≥ {near_sim} (weighted) of the project's 40 key terms (terms weighted by rarity across all project READMEs).",
         "Risk bands: 🔴 ≥3 near-misses or max recent sim ≥ 0.30 · 🟡 1–2 near-misses · 🟢 none. Similarity is lexical, so **read the flagged papers**; it cannot judge whether they close the gap.", "",
         "| risk | project | papers | ≥" + str(near_year) + " | near-misses | max recent sim | top recent near-miss |", "|---|---|---|---|---|---|---|"]
    for r in results:
        n = len(r["near_misses"])
        band = "🔴" if (n >= 3 or r["max_recent_sim"] >= 0.30) else ("🟡" if n else "🟢")
        top = r["near_misses"][0] if r["near_misses"] else None
        L.append(f"| {band} | [`{r['slug']}`](results/{r['slug']}.md) | {r['n_papers']} | {r['n_recent']} | {n} | {r['max_recent_sim']:.2f} | "
                 + (f"{top['year']} {top['title'][:90]}" if top else "") + " |")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    (OUT / "summary.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "top"} for r in results], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--backends", default="europepmc,pubmed,arxiv,crossref,openalex,s2")
    ap.add_argument("--since", type=int, default=2018)
    ap.add_argument("--near-year", type=int, default=2024)
    ap.add_argument("--near-sim", type=float, default=0.18)
    ap.add_argument("--no-cite-walk", action="store_true")
    ap.add_argument("--workers", type=int, default=3, help="projects processed concurrently")
    a = ap.parse_args()
    backends = [b for b in a.backends.split(",") if b in BACKENDS]
    backends = [b for b in backends if not ((b == "openalex" and not os.environ.get("OPENALEX_API_KEY")) or (b == "s2" and not os.environ.get("S2_API_KEY")))]
    queries = json.loads(QUERIES_FILE.read_text()) if QUERIES_FILE.exists() else {}
    slugs = a.slugs or sorted(d.name for d in PROJECTS.iterdir() if d.is_dir())
    print(f"backends: {backends}; projects: {len(slugs)}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(check_project, s, queries.get(s, []), backends, a.since, not a.no_cite_walk, a.near_year, a.near_sim): s for s in slugs}
        for f in as_completed(futs):
            s = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {s}: {e}", flush=True)
                continue
            write_result(r, a.near_year)
            results.append(r)
            print(f"{len(results):3d}/{len(slugs)} {s:45s} papers={r['n_papers']:3d} near={len(r['near_misses'])} maxsim={r['max_recent_sim']:.2f}", flush=True)
    # merge with earlier summary entries for slugs not run this time
    prev = OUT / "summary.json"
    if a.slugs and prev.exists():
        done = {r["slug"] for r in results}
        results += [r for r in json.loads(prev.read_text()) if r["slug"] not in done]
    write_summary(results, a.near_year, a.near_sim)
    print(f"wrote {OUT/'SUMMARY.md'}")


if __name__ == "__main__":
    main()

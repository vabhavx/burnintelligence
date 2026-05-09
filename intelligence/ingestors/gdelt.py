"""
BurnTheLies Intelligence — GDELT Ingestor
Pulls from GDELT GKG (Global Knowledge Graph) — 3M articles/day, 65+ languages, free.
Updates every 15 minutes.
"""

import asyncio
import aiohttp
import zipfile
import csv
import html
import io
import re
import hashlib
import logging
import ssl
import certifi
from datetime import datetime, timezone
from typing import AsyncGenerator

from intelligence.ingestors.retry import with_retry

logger = logging.getLogger("intelligence.gdelt")

THEME_CODE_RE = re.compile(r"\b[A-Z]+_[A-Z_0-9]+\b")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL | re.IGNORECASE)
MOJIBAKE_MARKERS = ("Â", "â€", "Ã©", "Ã¨", "Ã¢", "Ã ", "Ã±", "Ã¶", "Ã¼", "Ã¤", "Ã¡", "Ã­", "Ã³")

_STOPWORDS = {
    "the","a","an","and","or","but","of","to","in","on","at","by","for","with",
    "is","are","was","were","be","been","being","this","that","these","those",
    "it","its","as","from","into","than","then","so","if","not","no","do","does",
    "has","have","had","will","would","can","could","should","may","might",
}

_TLD_LANG = {
    "de": "de", "at": "de", "ch": "de",
    "fr": "fr", "be": "fr",
    "es": "es", "mx": "es", "ar": "es", "co": "es", "cl": "es", "pe": "es",
    "it": "it",
    "ru": "ru", "by": "ru",
    "cn": "zh", "tw": "zh", "hk": "zh",
    "jp": "ja", "kr": "ko",
    "br": "pt", "pt": "pt",
    "nl": "nl", "se": "sv", "no": "no", "fi": "fi", "dk": "da",
    "pl": "pl", "cz": "cs", "sk": "sk", "hu": "hu", "ro": "ro",
    "tr": "tr", "gr": "el", "il": "he", "sa": "ar", "ae": "ar", "eg": "ar",
    "ir": "fa", "th": "th", "vn": "vi", "id": "id",
    "in": "en", "uk": "en", "us": "en", "ca": "en", "au": "en", "nz": "en",
    "ie": "en", "za": "en", "ng": "en", "ke": "en", "ph": "en",
    "com": "unknown", "org": "unknown", "net": "unknown", "info": "unknown",
}

_INGEST_STATS = {
    "accepted": 0,
    "rejected_missing_url": 0,
    "rejected_empty_title": 0,
    "rejected_title_is_theme_codes": 0,
    "rejected_thin_content": 0,
    "rejected_mojibake": 0,
}


def get_ingest_stats() -> dict:
    return dict(_INGEST_STATS)


def _fix_mojibake(s: str) -> str:
    if not s:
        return s
    # GDELT GKG <PAGE_TITLE> is HTML-entity-encoded (e.g. &#x2013; for —,
    # &#x39D; for Ν). Unescape first so non-ASCII article titles read as
    # real text instead of literal "&#xNNN;" strings in the UI.
    if "&#" in s or "&amp;" in s or "&quot;" in s:
        try:
            s = html.unescape(s)
        except (TypeError, ValueError):
            pass
    if any(m in s for m in MOJIBAKE_MARKERS):
        try:
            return s.encode("latin-1", errors="strict").decode("utf-8", errors="replace")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def _is_printable_text(s: str) -> bool:
    if not s:
        return False
    good = sum(1 for c in s if c.isprintable() or c.isspace())
    return good / len(s) > 0.85


def _looks_like_theme_codes(s: str) -> bool:
    if not s:
        return True
    tokens = WORD_RE.findall(s)
    if not tokens:
        return True
    code_hits = len(THEME_CODE_RE.findall(s))
    return code_hits / len(tokens) > 0.30


def _is_real_content(title: str, content: str, url: str) -> tuple[bool, str]:
    if not url:
        return False, "missing_url"
    if not title or not title.strip():
        return False, "empty_title"
    if _looks_like_theme_codes(title):
        return False, "title_is_theme_codes"
    cleaned = THEME_CODE_RE.sub(" ", content or "")
    unique = {t.lower() for t in WORD_RE.findall(cleaned) if t.lower() not in _STOPWORDS and len(t) > 1}
    if len(unique) < 4:
        return False, "thin_content"
    return True, ""


def _extract_page_title(extras: str) -> str:
    if not extras:
        return ""
    m = PAGE_TITLE_RE.search(extras)
    if not m:
        return ""
    return m.group(1).strip()


def _lang_from_url(url: str, translated: bool) -> str:
    if translated:
        return "translated"  # not "en" — don't let translated articles bypass lang filter
    if not url:
        return "unknown"
    try:
        host = url.split("//", 1)[-1].split("/", 1)[0].lower()
        host = host.split(":", 1)[0]
        tld = host.rsplit(".", 1)[-1]
    except (IndexError, AttributeError):
        return "unknown"
    return _TLD_LANG.get(tld, "unknown")


def _reject(reason: str, url: str, title: str) -> None:
    key = f"rejected_{reason}"
    _INGEST_STATS[key] = _INGEST_STATS.get(key, 0) + 1
    logger.info(
        "GDELT reject [%s] url=%r title=%r",
        reason, (url or "")[:120], (title or "")[:120],
    )

# Optional: requests counter — no-op if prometheus_client missing
try:
    from prometheus_client import Counter as _PromCounter  # type: ignore

    gdelt_requests_total = _PromCounter(
        "gdelt_requests_total",
        "Total GDELT HTTP requests issued, labelled by endpoint and outcome.",
        labelnames=("endpoint", "outcome"),
    )
except Exception:  # pragma: no cover
    gdelt_requests_total = None


def _record_request(endpoint: str, outcome: str) -> None:
    if gdelt_requests_total is not None:
        try:
            gdelt_requests_total.labels(endpoint=endpoint, outcome=outcome).inc()
        except Exception:
            pass

# GDELT GKG v2 — master file lists all 15-minute update CSVs
GDELT_GKG_LASTUPDATE = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GDELT_GKG_TRANSLATION_LASTUPDATE = "http://data.gdeltproject.org/gdeltv2/lastupdate-translation.txt"

# GDELT Event mentions — alternative lighter-weight feed
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Themes that signal narrative manipulation / geopolitical tension
HIGH_SIGNAL_THEMES = {
    "INFORMATION_WARFARE", "DISINFORMATION", "PROPAGANDA",
    "MEDIA_CENSORSHIP", "CYBER_ATTACK", "ELECTION",
    "PROTEST", "MILITARY", "SANCTIONS", "COUP",
    "POLITICAL_TURMOIL", "CONSPIRACY", "CORRUPTION",
    "WEAPONIZED_NARRATIVE", "INFLUENCE_OPERATION",
    "TAX_FNCACT", "CRISISLEX_T03_DEAD",
}

# High-signal CAMEO event codes
CAMEO_CODES_CONFLICT = {"14", "15", "17", "18", "19", "20"}  # Protest, Force, Coerce
CAMEO_CODES_DIPLOMACY = {"03", "04", "05", "06"}  # Cooperation signals


async def fetch_gdelt_doc_api(session: aiohttp.ClientSession,
                               query: str = "",
                               mode: str = "ArtList",
                               timespan: str = "15min",
                               max_records: int = 250,
                               sourcelang: str = "") -> list[dict]:
    """
    Use GDELT DOC 2.0 API for targeted article retrieval.
    Free, no auth needed, returns structured JSON.
    """
    params = {
        "query": query or "disinformation OR propaganda OR \"influence operation\" OR \"information warfare\" OR manipulation",
        "mode": mode,
        "format": "json",
        "timespan": timespan,
        "maxrecords": str(max_records),
        "sort": "DateDesc",
    }
    if sourcelang:
        params["sourcelang"] = sourcelang

    _record_request("doc_api", "attempt")
    try:
        async with session.get(GDELT_DOC_API, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                logger.warning(f"GDELT DOC API returned {resp.status}")
                _record_request("doc_api", f"http_{resp.status}")
                return []
            data = await resp.json(content_type=None)
            articles = data.get("articles", [])
            logger.info(f"GDELT DOC API returned {len(articles)} articles")
            _record_request("doc_api", "ok")
            return articles
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        _record_request("doc_api", "error")
        raise


async def fetch_gdelt_gkg_latest(session: aiohttp.ClientSession,
                                  translation: bool = False) -> list[dict]:
    """
    Fetch the latest GKG 15-minute update file.
    Returns parsed records with themes and metadata.
    """
    url = GDELT_GKG_TRANSLATION_LASTUPDATE if translation else GDELT_GKG_LASTUPDATE
    endpoint = "gkg_translation" if translation else "gkg"

    _record_request(endpoint, "attempt")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            text = await resp.text()

        # Parse the lastupdate file — format: size hash url (3 entries: export, mentions, gkg)
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        gkg_line = [l for l in lines if "gkg" in l.lower()]
        if not gkg_line:
            logger.warning("No GKG file found in lastupdate")
            _record_request(endpoint, "no_file")
            return []

        gkg_url = gkg_line[0].split()[-1]  # URL is last field
        logger.info(f"Fetching GKG: {gkg_url}")

        async with session.get(gkg_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            data = await resp.read()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        _record_request(endpoint, "error")
        raise

    # GKG files are zipped CSVs — parsing happens outside the network try
    # block so transient HTTP errors are retryable but parse errors aren't.
    records = []
    rows_total = 0
    rows_skipped = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8', errors='replace'), delimiter='\t')
                    for row in reader:
                        rows_total += 1
                        try:
                            record = _parse_gkg_row(row)
                            if record and _is_high_signal(record):
                                record["translated"] = translation
                                records.append(record)
                        except (IndexError, ValueError, AttributeError):
                            rows_skipped += 1
                            continue
    except (zipfile.BadZipFile, csv.Error) as e:
        logger.error(f"GKG parse error: {e}")
        _record_request(endpoint, "parse_error")
        return []

    if rows_skipped:
        logger.warning(
            f"GKG: skipped {rows_skipped}/{rows_total} unparseable rows "
            f"(translation={translation})"
        )
    logger.info(f"GKG: {len(records)} high-signal records from latest update "
                f"(rows={rows_total})")
    _record_request(endpoint, "ok")
    return records


def _parse_gkg_row(row: list) -> dict | None:
    """Parse a GKG v2 tab-separated row into a structured dict."""
    if len(row) < 20:
        return None

    return {
        "gkg_id": row[0] if len(row) > 0 else "",
        "date": row[1] if len(row) > 1 else "",
        "source": row[3] if len(row) > 3 else "",
        "url": row[4] if len(row) > 4 else "",
        "themes": row[7].split(";") if len(row) > 7 and row[7] else [],
        "locations": row[9] if len(row) > 9 else "",   # V1Locations (# format, for reference)
        "persons": row[11] if len(row) > 11 else "",    # V1Persons (semicolon-separated names)
        "organizations": row[13] if len(row) > 13 else "",  # V1Organizations
        "tone": _parse_tone(row[15]) if len(row) > 15 else {},
        "gcam": row[17] if len(row) > 17 else "",  # Global Content Analysis Measures
        "sharing_image": row[19] if len(row) > 19 else "",
        "extras": row[-1] if len(row) > 20 else "",
    }


def _parse_tone(tone_str: str) -> dict:
    """Parse GDELT tone field: tone,positive,negative,polarity,activity,self,group"""
    parts = tone_str.split(",")
    if len(parts) < 6:
        return {}
    try:
        return {
            "tone": float(parts[0]),
            "positive": float(parts[1]),
            "negative": float(parts[2]),
            "polarity": float(parts[3]),
            "activity_density": float(parts[4]),
            "self_reference": float(parts[5]),
        }
    except (ValueError, IndexError):
        return {}


def _is_high_signal(record: dict) -> bool:
    """Filter for records likely related to narrative manipulation."""
    themes = set(record.get("themes", []))
    # Check theme overlap
    if themes & HIGH_SIGNAL_THEMES:
        return True
    # Check for high negative tone (potential manipulation indicator)
    tone = record.get("tone", {})
    if tone.get("polarity", 0) > 5 and tone.get("activity_density", 0) > 3:
        return True
    return False


async def ingest_cycle(db_conn, embed_fn=None) -> dict:
    """
    Run one full GDELT ingestion cycle.
    Returns stats dict.
    """
    from intelligence.db import insert_post

    stats = {
        "articles_fetched": 0, "articles_stored": 0, "duplicates": 0,
        "attempts": 0, "retries_used": 0, "last_error": None,
    }

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:
        # Parallel fetch: DOC API (targeted) + GKG (broad), each with retry.
        async def _doc():
            return await fetch_gdelt_doc_api(session, timespan="15min", max_records=250)

        async def _gkg():
            return await fetch_gdelt_gkg_latest(session, translation=False)

        async def _gkg_trans():
            return await fetch_gdelt_gkg_latest(session, translation=True)

        results = await asyncio.gather(
            with_retry(_doc, name="gdelt.doc_api"),
            with_retry(_gkg, name="gdelt.gkg"),
            with_retry(_gkg_trans, name="gdelt.gkg_translation"),
            return_exceptions=True,
        )

        unpacked = []
        for r in results:
            if isinstance(r, BaseException):
                stats["attempts"] += 1
                stats["last_error"] = f"{type(r).__name__}: {r}"
                unpacked.append(None)
                continue
            value, attempts, retries_used, last_error = r
            stats["attempts"] += attempts
            stats["retries_used"] += retries_used
            if last_error:
                stats["last_error"] = last_error
            unpacked.append(value if value is not None else [])

        doc_articles, gkg_records, gkg_trans = unpacked

        # Process DOC API articles
        if isinstance(doc_articles, list):
            for article in doc_articles:
                stats["articles_fetched"] += 1
                title = _fix_mojibake(article.get("title", ""))
                url = article.get("url", "")

                if title and not _is_printable_text(title):
                    _reject("mojibake", url, title)
                    continue

                ok, reason = _is_real_content(title, title, url)
                if not ok:
                    _reject(reason, url, title)
                    continue

                lang = article.get("language", "English")
                post_id = insert_post(
                    db_conn,
                    source="gdelt_doc",
                    content=title,
                    title=title,
                    url=url,
                    language=_normalize_lang(lang),
                    published_at=article.get("seendate", ""),
                    metadata={
                        "domain": article.get("domain", ""),
                        "socialimage": article.get("socialimage", ""),
                        "seendate": article.get("seendate", ""),
                    },
                )
                if post_id:
                    stats["articles_stored"] += 1
                    _INGEST_STATS["accepted"] += 1
                else:
                    stats["duplicates"] += 1

        # Process GKG records
        for gkg_list in [gkg_records, gkg_trans]:
            if not isinstance(gkg_list, list):
                continue
            for record in gkg_list:
                stats["articles_fetched"] += 1
                url = record.get("url", "")
                translated = bool(record.get("translated"))

                title = _fix_mojibake(_extract_page_title(record.get("extras", "")))

                if title and not _is_printable_text(title):
                    _reject("mojibake", url, title)
                    continue

                ok, reason = _is_real_content(title, title, url)
                if not ok:
                    _reject(reason, url, title)
                    continue

                gkg_date = record.get("date", "")
                published_at = ""
                if len(gkg_date) == 14:
                    try:
                        published_at = datetime(
                            int(gkg_date[:4]), int(gkg_date[4:6]), int(gkg_date[6:8]),
                            int(gkg_date[8:10]), int(gkg_date[10:12]), int(gkg_date[12:14]),
                            tzinfo=timezone.utc,
                        ).isoformat()
                    except ValueError:
                        published_at = ""

                post_id = insert_post(
                    db_conn,
                    source="gdelt_gkg",
                    content=title,
                    title=title,
                    url=url,
                    language=_lang_from_url(url, translated),
                    published_at=published_at,
                    metadata={
                        "themes": record.get("themes", []),
                        "tone": record.get("tone", {}),
                        "persons": record.get("persons", ""),
                        "organizations": record.get("organizations", ""),
                        "locations": record.get("locations", ""),
                        "translated": translated,
                    },
                )
                if post_id:
                    stats["articles_stored"] += 1
                    _INGEST_STATS["accepted"] += 1
                else:
                    stats["duplicates"] += 1

    logger.info(f"GDELT cycle complete: {stats}")
    return stats


def _normalize_lang(lang: str) -> str:
    """Normalize GDELT language names to ISO codes."""
    mapping = {
        "English": "en", "Spanish": "es", "French": "fr",
        "German": "de", "Chinese": "zh", "Russian": "ru",
        "Arabic": "ar", "Hindi": "hi", "Portuguese": "pt",
        "Japanese": "ja", "Korean": "ko", "Turkish": "tr",
    }
    return mapping.get(lang, lang.lower()[:2] if lang else "en")

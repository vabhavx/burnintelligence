"""
BurnTheLies Intelligence — Bluesky Firehose Ingestor
AT Protocol Jetstream — fully open, zero cost, 40M+ users.
Monitors for narrative-relevant posts in real-time.
"""

import asyncio
import json
import logging
import os
import re
import ssl
import aiohttp
import certifi
from datetime import datetime, timezone
from typing import Optional

from intelligence.ingestors.retry import with_retry

logger = logging.getLogger("intelligence.bluesky")

# Bluesky Jetstream — public WebSocket endpoint (no auth needed for public firehose)
JETSTREAM_URL = "wss://jetstream1.us-east.bsky.network/subscribe"

# High-specificity keywords — one match alone is sufficient evidence
SIGNAL_KEYWORDS_HIGH = {
    "psyop", "false flag", "info war", "bot network", "astroturf",
    "influence operation", "media manipulation", "whistleblower",
    "who benefits", "follow the money", "cui bono",
    "state-sponsored", "troll farm", "sockpuppet", "coordinated inauthentic",
}

# Low-specificity keywords — require TWO or more from this set to match
SIGNAL_KEYWORDS_LOW = {
    "disinformation", "propaganda", "narrative", "manipulation",
    "coordinated", "leaked", "exposed", "classified", "censorship", "cover up",
}

_PATTERN_HIGH = re.compile(
    "|".join(re.escape(kw) for kw in SIGNAL_KEYWORDS_HIGH),
    re.IGNORECASE
)
_PATTERN_LOW = re.compile(
    "|".join(re.escape(kw) for kw in SIGNAL_KEYWORDS_LOW),
    re.IGNORECASE
)

# Backward-compat alias used elsewhere
SIGNAL_KEYWORDS = SIGNAL_KEYWORDS_HIGH | SIGNAL_KEYWORDS_LOW
SIGNAL_PATTERN = _PATTERN_HIGH


async def connect_firehose(db_conn, max_posts: int = 0,
                           duration_seconds: int = 0) -> dict:
    """
    Connect to Bluesky Jetstream firehose with bounded session timeout
    and retry on transient connection failures (max 2 retries — reconnect
    is expensive).
    """
    stats = {
        "received": 0, "matched": 0, "stored": 0, "errors": 0,
        "attempts": 0, "retries_used": 0, "last_error": None,
    }

    async def _run_once():
        await _firehose_session(db_conn, stats, max_posts, duration_seconds)

    _, attempts, retries_used, last_error = await with_retry(
        _run_once,
        retries=2,
        base_delay=2.0,
        max_delay=30.0,
        name="bluesky.firehose",
    )
    stats["attempts"] = attempts
    stats["retries_used"] = retries_used
    if last_error:
        stats["last_error"] = last_error

    logger.info(f"Bluesky firehose session complete: {stats}")
    return stats


async def _firehose_session(db_conn, stats: dict, max_posts: int,
                            duration_seconds: int) -> None:
    """One firehose connection attempt. Raises on transient network errors
    so the outer retry can re-connect."""
    from intelligence.db import insert_post

    start_time = asyncio.get_event_loop().time()
    total_timeout = int(os.getenv("BLUESKY_FIREHOSE_TOTAL_TIMEOUT", "360"))

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:
        params = {
            "wantedCollections": "app.bsky.feed.post",
        }
        async with session.ws_connect(
            JETSTREAM_URL,
            params=params,
            heartbeat=30,
            timeout=aiohttp.ClientTimeout(total=total_timeout, sock_read=60),
        ) as ws:
            logger.info("Connected to Bluesky Jetstream firehose")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                        stats["received"] += 1

                        post = _extract_post(event)
                        if post and _is_signal_relevant(post):
                            stats["matched"] += 1

                            post_id = insert_post(
                                db_conn,
                                source="bluesky",
                                content=post["text"],
                                author=post["author"],
                                language=post.get("language", "en"),
                                published_at=post.get("created_at", ""),
                                url=post.get("url", ""),
                                metadata={
                                    "reply_to": post.get("reply_to"),
                                    "has_embed": post.get("has_embed", False),
                                    "langs": post.get("langs", []),
                                },
                            )
                            if post_id:
                                stats["stored"] += 1

                    except json.JSONDecodeError:
                        stats["errors"] += 1
                    except (KeyError, TypeError, ValueError) as e:
                        stats["errors"] += 1
                        if stats["errors"] % 100 == 0:
                            logger.warning(f"Firehose parse error #{stats['errors']}: {e}")

                elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                    logger.warning(f"WebSocket {msg.type}")
                    break

                # Check exit conditions
                if max_posts and stats["stored"] >= max_posts:
                    break
                if duration_seconds:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration_seconds:
                        break

                # Log progress periodically
                if stats["received"] % 10000 == 0 and stats["received"] > 0:
                    logger.info(
                        f"Firehose: {stats['received']} received, "
                        f"{stats['matched']} matched, {stats['stored']} stored"
                    )


def _extract_post(event: dict) -> Optional[dict]:
    """Extract post data from a Jetstream event."""
    if event.get("kind") != "commit":
        return None

    commit = event.get("commit", {})
    if commit.get("operation") != "create":
        return None
    if commit.get("collection") != "app.bsky.feed.post":
        return None

    record = commit.get("record", {})
    text = record.get("text", "")
    if not text or len(text) < 10:
        return None

    did = event.get("did", "")
    rkey = commit.get("rkey", "")

    return {
        "text": text,
        "author": did,
        "created_at": record.get("createdAt", ""),
        "langs": record.get("langs", []),
        "language": record.get("langs", ["en"])[0] if record.get("langs") else "en",
        "reply_to": record.get("reply", {}).get("parent", {}).get("uri", ""),
        "has_embed": "embed" in record,
        "url": f"https://bsky.app/profile/{did}/post/{rkey}" if did and rkey else "",
    }


def _is_signal_relevant(post: dict) -> bool:
    """Check if a post matches narrative manipulation signals.

    Requires either one high-specificity keyword OR two distinct low-specificity
    keywords, plus a minimum text length to filter short reactions/noise.
    """
    text = post.get("text", "")
    if len(text) < 30:
        return False
    if _PATTERN_HIGH.search(text):
        return True
    low_matches = _PATTERN_LOW.findall(text.lower())
    return len(set(low_matches)) >= 2


async def sample_firehose(duration_seconds: int = 30) -> dict:
    """
    Quick sample of the firehose for testing.
    Returns stats without storing.
    """
    stats = {"received": 0, "text_posts": 0, "signal_matched": 0}
    start = asyncio.get_event_loop().time()

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:
        async with session.ws_connect(
            JETSTREAM_URL,
            params={"wantedCollections": "app.bsky.feed.post"},
            heartbeat=30,
        ) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    stats["received"] += 1
                    try:
                        event = json.loads(msg.data)
                        post = _extract_post(event)
                        if post:
                            stats["text_posts"] += 1
                            if _is_signal_relevant(post):
                                stats["signal_matched"] += 1
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                        logger.warning(f"sample_firehose: skipped malformed event ({type(e).__name__}: {e})")

                if asyncio.get_event_loop().time() - start > duration_seconds:
                    break

    return stats

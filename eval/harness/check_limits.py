"""check_limits.py

Probe a Groq API key and report:
  - Per-minute token budget (from response headers)
  - Daily token budget (parsed from 429 error body when limit is hit)
  - Estimated reset time for the daily limit (midnight UTC)

Usage (from project root):
    python eval/harness/check_limits.py <api_key>
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta


GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
PROBE_PAYLOAD = json.dumps({
    "model": "llama-3.3-70b-versatile",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 1,
}).encode()


def _time_until_midnight_utc() -> str:
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    delta = midnight - now
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def check(api_key: str) -> None:
    req = urllib.request.Request(
        GROQ_CHAT_URL,
        data=PROBE_PAYLOAD,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "python-httpx/0.25.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            headers = dict(resp.headers)
            limit_rpm   = headers.get("x-ratelimit-limit-tokens", "?")
            remain_rpm  = headers.get("x-ratelimit-remaining-tokens", "?")
            reset_rpm   = headers.get("x-ratelimit-reset-tokens", "?")

            print(f"  Status            : OK")
            print(f"  Tokens/min  limit : {limit_rpm}")
            print(f"  Tokens/min  left  : {remain_rpm}")
            print(f"  Tokens/min  reset : {reset_rpm}")
            print(f"  Daily limit reset : ~{_time_until_midnight_utc()} (midnight UTC)")

    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")

        if e.code == 429:
            tpd_match = re.search(
                r"Limit\s+([\d,]+),\s+Used\s+([\d,]+),\s+Requested\s+([\d,]+)",
                body,
            )
            retry_match = re.search(r"try again in ([\w\d]+)", body)

            print(f"  Status            : LIMIT HIT (429)")
            if tpd_match:
                limit     = tpd_match.group(1).replace(",", "")
                used      = tpd_match.group(2).replace(",", "")
                requested = tpd_match.group(3).replace(",", "")
                remaining = int(limit) - int(used)
                print(f"  Daily token limit : {int(limit):,}")
                print(f"  Daily tokens used : {int(used):,}  ({100*int(used)//int(limit)}%)")
                print(f"  Daily tokens left : {remaining:,}")
                print(f"  Last request size : {int(requested):,} tokens")
            if retry_match:
                print(f"  Retry after       : {retry_match.group(1)}")
            print(f"  Daily limit reset : ~{_time_until_midnight_utc()} (midnight UTC)")
        else:
            print(f"  Status            : ERROR {e.code}")
            print(f"  Body              : {body[:200]}")

    except Exception as exc:
        print(f"  Status            : UNREACHABLE ({exc})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/harness/check_limits.py <api_key>")
        sys.exit(1)

    check(sys.argv[1])

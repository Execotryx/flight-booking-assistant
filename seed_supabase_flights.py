"""Seed initial flight inventory into Supabase using the REST API.

This script loads a dedicated `.seeding-env` file (via python-dotenv)
and supports environment variable overrides.

Environment overrides:
- SUPABASE_URL
- SUPABASE_KEY
- SUPABASE_TABLE (default: flights)
- SUPABASE_LOOKUP_TABLE (default: city_code_lookup)

Example:
    e:/sources/python/llm-engineering/day4/flight-ticket-assistant/.venv/Scripts/python.exe seed_supabase_flights.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

from dotenv import dotenv_values, load_dotenv

DEFAULT_SEEDING_ENV_FILE: str = ".seeding-env"
DEFAULT_TABLE: str = "flights"
DEFAULT_LOOKUP_TABLE: str = "city_code_lookup"

SEED_FLIGHTS: list[dict[str, object]] = [
    {
        "flight_id": "FL-1001",
        "airline": "SkyJet",
        "origin": "NYC",
        "destination": "LON",
        "date": "2026-03-20",
        "depart_time": "09:10",
        "arrive_time": "21:45",
        "base_fare_usd": 640,
        "seats_left": 4,
    },
    {
        "flight_id": "FL-1002",
        "airline": "AeroWays",
        "origin": "NYC",
        "destination": "LON",
        "date": "2026-03-20",
        "depart_time": "18:35",
        "arrive_time": "07:05",
        "base_fare_usd": 560,
        "seats_left": 7,
    },
    {
        "flight_id": "FL-2001",
        "airline": "PacificAir",
        "origin": "LAX",
        "destination": "NRT",
        "date": "2026-03-22",
        "depart_time": "11:40",
        "arrive_time": "16:10",
        "base_fare_usd": 790,
        "seats_left": 5,
    },
    {
        "flight_id": "FL-3001",
        "airline": "EuroConnect",
        "origin": "LON",
        "destination": "CDG",
        "date": "2026-03-18",
        "depart_time": "14:15",
        "arrive_time": "16:20",
        "base_fare_usd": 120,
        "seats_left": 9,
    },
]

CITY_TO_CODE: dict[str, str] = {
    "new york": "NYC",
    "nyc": "NYC",
    "london": "LON",
    "lon": "LON",
    "paris": "CDG",
    "cdg": "CDG",
    "los angeles": "LAX",
    "lax": "LAX",
    "tokyo": "NRT",
    "nrt": "NRT",
}

SEED_CITY_CODE_LOOKUP: list[dict[str, object]] = [
    {"city_name": city_name, "iata_code": iata_code}
    for city_name, iata_code in CITY_TO_CODE.items()
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed flights into Supabase table")
    parser.add_argument(
        "--table",
        default=os.getenv("SUPABASE_TABLE", DEFAULT_TABLE),
        help="Target Supabase table name (default: flights).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows in the table before seeding.",
    )
    parser.add_argument(
        "--lookup-table",
        default=os.getenv("SUPABASE_LOOKUP_TABLE", DEFAULT_LOOKUP_TABLE),
        help="Target lookup table for city-to-code mappings (default: city_code_lookup).",
    )
    parser.add_argument(
        "--skip-lookup",
        action="store_true",
        help="Skip seeding city-to-code lookup table.",
    )
    return parser.parse_args()


def _request(
    *,
    method: str,
    url: str,
    api_key: str,
    body: list[dict[str, object]] | None = None,
) -> tuple[int, str]:
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url=url, data=data, method=method)
    req.add_header("apikey", api_key)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Prefer", "resolution=merge-duplicates,return=representation")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read().decode("utf-8")
            return resp.status, payload
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        return exc.code, error_body


def _delete_all(base_url: str, table: str, key_column: str, api_key: str) -> None:
    """Delete all rows from a table using an always-true PostgREST filter.

    Parameters:
        base_url: Supabase project URL.
        table: Target table name.
        key_column: Non-nullable key column used in `not.is.null` filter.
        api_key: Supabase API key.
    """
    url = (
        f"{base_url}/rest/v1/{urllib.parse.quote(table)}?"
        f"{urllib.parse.quote(key_column)}=not.is.null"
    )
    status, payload = _request(method="DELETE", url=url, api_key=api_key, body=None)
    if status >= 400:
        raise RuntimeError(f"Delete failed ({status}): {payload}")


def _upsert_seed(base_url: str, table: str, api_key: str) -> None:
    query = urllib.parse.urlencode({"on_conflict": "flight_id"})
    url = f"{base_url}/rest/v1/{urllib.parse.quote(table)}?{query}"
    status, payload = _request(method="POST", url=url, api_key=api_key, body=SEED_FLIGHTS)
    if status >= 400:
        raise RuntimeError(f"Seed failed ({status}): {payload}")

    print(f"Seeded {len(SEED_FLIGHTS)} flights into '{table}'.")
    if payload:
        print(payload)


def _upsert_city_code_lookup(base_url: str, table: str, api_key: str) -> None:
    """Upsert city-to-IATA lookup rows.

    Parameters:
        base_url: Supabase project URL.
        table: Lookup table name.
        api_key: Supabase API key.
    """
    query = urllib.parse.urlencode({"on_conflict": "city_name"})
    url = f"{base_url}/rest/v1/{urllib.parse.quote(table)}?{query}"
    status, payload = _request(
        method="POST",
        url=url,
        api_key=api_key,
        body=SEED_CITY_CODE_LOOKUP,
    )
    if status >= 400:
        raise RuntimeError(f"Lookup seed failed ({status}): {payload}")

    print(f"Seeded {len(SEED_CITY_CODE_LOOKUP)} city-code rows into '{table}'.")
    if payload:
        print(payload)


def main() -> int:
    # Load regular env first, then dedicated seeding env file.
    load_dotenv()
    load_dotenv(DEFAULT_SEEDING_ENV_FILE, override=True)

    args = _parse_args()

    seeding_file_values = dotenv_values(DEFAULT_SEEDING_ENV_FILE)

    base_url = str(
        os.getenv("SUPABASE_URL")
        or seeding_file_values.get("SUPABASE_URL")
        or ""
    ).rstrip("/")
    api_key = str(
        os.getenv("SUPABASE_KEY")
        or seeding_file_values.get("SUPABASE_KEY")
        or ""
    ).strip()

    if not base_url.startswith("http"):
        print(
            "SUPABASE_URL must be set in environment or .seeding-env and be a valid URL.",
            file=sys.stderr,
        )
        return 2

    if not api_key:
        print("SUPABASE_KEY is empty.", file=sys.stderr)
        return 2

    print(
        f"Target: {base_url} table={args.table} lookup_table={args.lookup_table}"
    )

    try:
        if args.replace:
            print("Deleting existing rows...")
            _delete_all(
                base_url=base_url,
                table=args.table,
                key_column="flight_id",
                api_key=api_key,
            )
            if not args.skip_lookup:
                _delete_all(
                    base_url=base_url,
                    table=args.lookup_table,
                    key_column="city_name",
                    api_key=api_key,
                )

        print("Seeding flights...")
        _upsert_seed(base_url=base_url, table=args.table, api_key=api_key)

        if not args.skip_lookup:
            print("Seeding city-code lookup...")
            _upsert_city_code_lookup(
                base_url=base_url,
                table=args.lookup_table,
                api_key=api_key,
            )

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "If this is an RLS/permission error, use a service role key or adjust table policies.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

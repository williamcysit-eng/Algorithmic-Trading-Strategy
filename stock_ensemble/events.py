"""Immutable source snapshots and conservative as-of views of public events.

No event is available before this collector first observes its source. Publication
metadata is retained separately; importing an old release never backdates data.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

KINDS = ("earnings_release", "earnings_schedule", "dividend_schedule", "consensus_snapshot")
SCHEMA = 1


def _now():
    return pd.Timestamp.now(tz="UTC")


def _timestamp(value, label):
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return stamp.tz_convert("UTC")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _publication(value, zone, observed):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        start = pd.Timestamp(value).tz_localize(ZoneInfo(zone))
        if start.date() > observed.tz_convert(zone).date():
            raise ValueError("Publication date cannot be after collection")
        bound = (start + pd.DateOffset(days=1)).tz_convert("UTC")
        return "day", bound
    stamp = _timestamp(value, "Publication timestamp")
    if stamp > observed:
        raise ValueError("Publication timestamp cannot be after collection")
    return "timestamp", stamp


@contextmanager
def _locked(store):
    store.mkdir(parents=True, exist_ok=True)
    lock = store / ".writer-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise ValueError(f"Another writer holds {lock}; inspect it before retrying") from error
    try:
        yield
    finally:
        lock.rmdir()


def load_ledger(store):
    """Verify source content, append history, revision order, and availability."""
    store = Path(store)
    path = store / "ledger.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA or not isinstance(payload.get("records"), list):
        raise ValueError("Unsupported event ledger schema")
    records = payload["records"]
    previous, versions, last_observed, identities = "", {}, None, {}
    for record in records:
        content = {key: value for key, value in record.items() if key != "record_sha256"}
        if _digest(content) != record.get("record_sha256"):
            raise ValueError("Event record checksum mismatch")
        if record["previous_record_sha256"] != previous:
            raise ValueError("Event append chain is broken")
        source_hash = record["source_sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ValueError("Malformed event source digest")
        source = store / "objects" / source_hash
        if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
            raise ValueError("Event source snapshot checksum mismatch")
        event = record["event_id"]
        identity = (record["ticker"], record["kind"])
        if event in identities and identities[event] != identity:
            raise ValueError("Event identity changed across revisions")
        identities[event] = identity
        expected = versions.get(event, 0) + 1
        if record["revision"] != expected:
            raise ValueError("Event revision sequence is broken")
        observed = _timestamp(record["observed_at"], "Collection timestamp")
        if last_observed is not None and observed < last_observed:
            raise ValueError("Event collection order runs backward")
        precision, publication_bound = _publication(
            record["published"], record["publication_zone"], observed
        )
        if record["publication_precision"] != precision:
            raise ValueError("Publication precision mismatch")
        available = _timestamp(record["available_at"], "Availability timestamp")
        if available != max(publication_bound, observed):
            raise ValueError("Event availability predates publication or collection")
        previous, versions[event], last_observed = record["record_sha256"], expected, observed
    return records


def observe_source(
    store,
    *,
    event_id,
    ticker,
    kind,
    source_url,
    published,
    document,
    publication_zone="America/New_York",
    fields=None,
):
    """Append a source revision; exact repeat imports retain original availability."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,159}", event_id):
        raise ValueError("Malformed event_id")
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]*", ticker) or kind not in KINDS:
        raise ValueError("Invalid event ticker or kind")
    url = urllib.parse.urlparse(source_url)
    if url.scheme != "https" or not url.netloc or url.username or url.password:
        raise ValueError("Event sources require a public HTTPS URL without credentials")
    if not isinstance(document, bytes) or not document:
        raise ValueError("Source document must contain original bytes")
    if fields is None:
        fields = {}
    if not isinstance(fields, dict):
        raise TypeError("Event fields must be a JSON object")
    # Validate metadata before touching the ledger or publishing an object.
    fields = json.loads(json.dumps(fields, allow_nan=False))
    observed = _timestamp(_now(), "Collection timestamp")
    precision, publication_bound = _publication(published, publication_zone, observed)
    source_hash = hashlib.sha256(document).hexdigest()
    source_fields = {
        "event_id": event_id,
        "ticker": ticker,
        "kind": kind,
        "source_url": source_url,
        "published": published,
        "publication_zone": publication_zone,
        "publication_precision": precision,
        "source_sha256": source_hash,
        "fields": fields,
    }
    store = Path(store)
    with _locked(store):
        records = load_ledger(store)
        old = [record for record in records if record["event_id"] == event_id]
        if old and (old[-1]["ticker"], old[-1]["kind"]) != (ticker, kind):
            raise ValueError("Existing event_id belongs to another ticker or kind")
        for record in old:
            if all(record[key] == value for key, value in source_fields.items()):
                return record
        if records and observed < pd.Timestamp(records[-1]["observed_at"]):
            raise ValueError("Collection clock runs backward")
        record = {
            **source_fields,
            "observed_at": observed.isoformat(),
            "available_at": max(publication_bound, observed).isoformat(),
            "revision": len(old) + 1,
            "previous_record_sha256": records[-1]["record_sha256"] if records else "",
        }
        record["record_sha256"] = _digest(record)
        objects = store / "objects"
        objects.mkdir(exist_ok=True)
        target = objects / source_hash
        if target.exists():
            if target.read_bytes() != document:
                raise ValueError("Conflicting source snapshot")
        else:
            with target.open("xb") as stream:
                stream.write(document)
        payload = {"schema_version": SCHEMA, "records": [*records, record]}
        temporary = store / ".ledger.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(store / "ledger.json")
    return record


def events_as_of(store, cutoff):
    cutoff = _timestamp(cutoff, "As-of cutoff")
    latest = {}
    for record in load_ledger(store):
        if pd.Timestamp(record["available_at"]) <= cutoff:
            latest[record["event_id"]] = record
    return [latest[key] for key in sorted(latest)]


def capture_url(url, user_agent):
    if not user_agent.strip():
        raise ValueError("A source-identifying user agent is required")
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("Capture requires HTTPS")
    if (
        urllib.parse.urlparse(url).hostname in ("sec.gov", "www.sec.gov", "data.sec.gov")
        and "@" not in user_agent
    ):
        raise ValueError("SEC capture requires an identifying user agent with contact email")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=20) as response:
        if urllib.parse.urlparse(response.geturl()).scheme != "https":
            raise ValueError("Source redirected outside HTTPS")
        document = response.read(10_000_001)
    if len(document) > 10_000_000:
        raise ValueError("Source exceeds the 10 MB capture limit")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "import"):
        command = commands.add_parser(name)
        command.add_argument("--store", default="data/events")
        command.add_argument("--event-id", required=True)
        command.add_argument("--ticker", required=True)
        command.add_argument("--kind", choices=KINDS, required=True)
        command.add_argument("--source-url", required=True)
        command.add_argument(
            "--published", required=True, help="Original timezone timestamp or date"
        )
        command.add_argument("--publication-zone", default="America/New_York")
        command.add_argument("--fields", type=Path, help="Optional metadata JSON object")
        if name == "capture":
            command.add_argument("--user-agent", default="STAT7008 academic source capture/1.0")
        else:
            command.add_argument("--document", type=Path, required=True)
    view = commands.add_parser("as-of")
    view.add_argument("--store", default="data/events")
    view.add_argument("--at", required=True, help="Timezone-aware information cutoff")
    args = parser.parse_args(argv)
    try:
        if args.command == "as-of":
            result = events_as_of(args.store, args.at)
        else:
            document = (
                capture_url(args.source_url, args.user_agent)
                if args.command == "capture"
                else args.document.read_bytes()
            )
            fields = json.loads(args.fields.read_text()) if args.fields else {}
            result = observe_source(
                args.store,
                event_id=args.event_id,
                ticker=args.ticker,
                kind=args.kind,
                source_url=args.source_url,
                published=args.published,
                document=document,
                publication_zone=args.publication_zone,
                fields=fields,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

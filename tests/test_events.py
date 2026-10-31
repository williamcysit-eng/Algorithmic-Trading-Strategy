"""Source availability, revision stability, idempotence, and tamper checks."""

import hashlib
import json

import pandas as pd
import pytest

from stock_ensemble import events


@pytest.fixture
def source():
    return {
        "event_id": "A-2025Q4-earnings",
        "ticker": "A",
        "kind": "earnings_release",
        "source_url": "https://example.org/issuer/release",
        "published": "2025-10-30",
        "publication_zone": "America/New_York",
        "document": b"Original issuer statement",
        "fields": {"status": "synthetic unit-test document; not market data"},
    }


def test_old_source_cannot_be_backdated_and_duplicates_are_idempotent(
    tmp_path, monkeypatch, source
):
    collected = pd.Timestamp("2026-10-31T16:00:00Z")
    monkeypatch.setattr(events, "_now", lambda: collected)
    record = events.observe_source(tmp_path, **source)
    assert record["available_at"] == collected.isoformat()
    assert record["publication_precision"] == "day"
    assert events.events_as_of(tmp_path, "2025-11-01T00:00:00Z") == []
    assert events.events_as_of(tmp_path, collected) == [record]
    before = (tmp_path / "ledger.json").read_bytes()
    monkeypatch.setattr(events, "_now", lambda: collected + pd.Timedelta(days=1))
    assert events.observe_source(tmp_path, **source) == record
    assert (tmp_path / "ledger.json").read_bytes() == before
    assert record["source_sha256"] == hashlib.sha256(source["document"]).hexdigest()


def test_corrections_append_without_rewriting_previous_views(tmp_path, monkeypatch, source):
    first = pd.Timestamp("2026-10-31T16:00:00Z")
    monkeypatch.setattr(events, "_now", lambda: first)
    original = events.observe_source(tmp_path, **source)
    monkeypatch.setattr(events, "_now", lambda: first + pd.Timedelta(hours=1))
    correction = events.observe_source(tmp_path, **{**source, "document": b"Corrected statement"})
    assert correction["revision"] == 2
    assert correction["previous_record_sha256"] == original["record_sha256"]
    assert events.events_as_of(tmp_path, first + pd.Timedelta(minutes=1)) == [original]
    assert events.events_as_of(tmp_path, first + pd.Timedelta(hours=1)) == [correction]
    assert len(events.load_ledger(tmp_path)) == 2
    # Reimporting the original bytes does not roll the latest version backward.
    events.observe_source(tmp_path, **source)
    assert events.events_as_of(tmp_path, first + pd.Timedelta(hours=2)) == [correction]


def test_date_only_publication_is_conservative_and_timezone_required(tmp_path, monkeypatch, source):
    collected = pd.Timestamp("2026-10-31T16:00:00Z")
    monkeypatch.setattr(events, "_now", lambda: collected)
    record = events.observe_source(tmp_path, **{**source, "published": "2026-10-31"})
    assert pd.Timestamp(record["available_at"]) == pd.Timestamp("2026-11-01T04:00:00Z")
    assert events.events_as_of(tmp_path, collected) == []
    assert events.events_as_of(tmp_path, "2026-11-01T04:00:00Z") == [record]
    with pytest.raises(ValueError, match="include a timezone"):
        events.events_as_of(tmp_path, "2026-11-01")
    with pytest.raises(ValueError, match="after collection"):
        events.observe_source(tmp_path, **{**source, "published": "2026-11-02"})
    with pytest.raises(ValueError, match="include a timezone"):
        events.observe_source(tmp_path, **{**source, "published": "2025-10-30T16:00:00"})


def test_snapshot_and_metadata_tampering_fail_closed(tmp_path, monkeypatch, source):
    monkeypatch.setattr(events, "_now", lambda: pd.Timestamp("2026-10-31T16:00:00Z"))
    record = events.observe_source(tmp_path, **source)
    snapshot = tmp_path / "objects" / record["source_sha256"]
    snapshot.write_bytes(b"Changed in place")
    with pytest.raises(ValueError, match="snapshot checksum"):
        events.load_ledger(tmp_path)
    snapshot.write_bytes(source["document"])
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    ledger["records"][0]["available_at"] = "2020-01-01T00:00:00Z"
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="record checksum"):
        events.load_ledger(tmp_path)


def test_event_identity_finite_metadata_clock_and_lock_are_validated(tmp_path, monkeypatch, source):
    collected = pd.Timestamp("2026-10-31T16:00:00Z")
    monkeypatch.setattr(events, "_now", lambda: collected)
    events.observe_source(tmp_path, **source)
    with pytest.raises(ValueError, match="another ticker"):
        events.observe_source(tmp_path, **{**source, "ticker": "B"})
    with pytest.raises(ValueError):
        events.observe_source(tmp_path, **{**source, "fields": {"value": float("nan")}})
    monkeypatch.setattr(events, "_now", lambda: collected - pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="clock runs backward"):
        events.observe_source(tmp_path, **{**source, "document": b"New bytes"})
    monkeypatch.setattr(events, "_now", lambda: collected)
    (tmp_path / ".writer-lock").mkdir()
    with pytest.raises(ValueError, match="Another writer"):
        events.observe_source(tmp_path, **source)
    (tmp_path / ".writer-lock").rmdir()
    with pytest.raises(ValueError, match="contact email"):
        events.capture_url("https://www.sec.gov/Archives/example", "academic capture")


def test_import_and_asof_cli_use_same_validated_ledger(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(events, "_now", lambda: pd.Timestamp("2026-10-31T16:00:00Z"))
    document = tmp_path / "release.txt"
    document.write_bytes(b"Synthetic unit-test source")
    store = tmp_path / "events"
    assert (
        events.main(
            [
                "import",
                "--store",
                str(store),
                "--event-id",
                "TEST-earnings",
                "--ticker",
                "TEST",
                "--kind",
                "earnings_release",
                "--source-url",
                "https://example.org/release",
                "--published",
                "2025-10-30",
                "--document",
                str(document),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert events.main(["as-of", "--store", str(store), "--at", "2025-10-31T00:00:00Z"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert events.main(["as-of", "--store", str(store), "--at", "2026-10-31T16:00:00Z"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1

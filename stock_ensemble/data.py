"""Public-data ingestion; filing dates, not fiscal dates, control earnings availability.

The fixed, present-day universe is survivor-selected. SEC seasonal profit changes are
an accounting surprise proxy, not analyst-consensus earnings surprises. Raw source
snapshots and their hashes make a downloaded pilot reproducible, not survivorship-free.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["date", "ticker", "close", "adj_close", "volume", "dividend", "split"]
EARNINGS_COLUMNS = [
    "ticker",
    "filed_date",
    "fiscal_end",
    "surprise",
    "seasonal_change",
    "source_accn",
    "source_concept",
]
UNIVERSE_COLUMNS = ["ticker", "cik", "name", "sector"]
_CACHE_VERSION = 1
_NY = ZoneInfo("America/New_York")
_LIMITATIONS = [
    "Present-day, survivor-selected US equity universe; no historical membership or delisted stocks.",
    "Yahoo and SEC are revisable public sources, not an institutional point-in-time archive.",
    "SEC filing dates conservatively lag actual earnings releases; signals require strictly earlier filings.",
    (
        "Surprise is standardized seasonal profit growth divided by assets, not consensus forecast error; "
        "NetIncomeLoss and earnings available to common stockholders are standardized separately."
    ),
    "SEC concept/coverage gaps produce missing earnings signals, never fabricated fundamentals.",
    (
        "Yahoo historical nominal prices, volumes and dividends are reconstructed from subsequent splits; "
        "unreported corporate actions and vendor errors cannot be independently excluded."
    ),
    "Price-only returns exclude dividends; adjusted-close returns are a distinct total-return alternative.",
]


def _date(value, label):
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError(f"{label} must be a timezone-naive calendar date: {value!r}")
    return result


def load_universe(path) -> pd.DataFrame:
    """Load an explicit equity research universe without dropping any rows."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(UNIVERSE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Universe is missing columns: {sorted(missing)}")
    frame = frame[UNIVERSE_COLUMNS].copy()
    for column in UNIVERSE_COLUMNS:
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Universe has blank {column}")
    frame["ticker"] = frame["ticker"].str.upper()
    if not frame["ticker"].str.fullmatch(r"[A-Z][A-Z0-9.\-]*").all():
        raise ValueError("Universe has malformed Yahoo equity tickers")
    if frame.empty or frame["ticker"].duplicated().any():
        raise ValueError("Universe must contain unique, nonempty tickers")
    if not frame["cik"].str.fullmatch(r"\d{1,10}").all():
        raise ValueError("CIK must be a positive integer with at most ten digits")
    frame["cik"] = frame["cik"].map(lambda value: str(int(value)).zfill(10))
    if frame["cik"].eq("0000000000").any():
        raise ValueError("CIK must be positive")
    return frame.sort_values("ticker").reset_index(drop=True)


def _facts(payload, concept, duration):
    raw = (
        payload.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {}).get("USD", [])
    )
    facts = []
    for item in raw:
        if item.get("form") not in {"10-K", "10-Q", "10-K/A", "10-Q/A"}:
            continue
        # Some issuers tag transitional/instant income contexts. Without a
        # duration they cannot identify quarterly earnings; retain them in the
        # raw snapshot, but do not treat them as usable quarterly observations.
        if duration and not item.get("start"):
            continue
        try:
            filed = _date(item["filed"], "SEC filed date")
            end = _date(item["end"], "SEC fiscal end")
            start = _date(item["start"], "SEC fiscal start") if duration else None
            value = float(item["val"])
            accession = str(item["accn"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed SEC {concept} fact: {item!r}") from exc
        if not math.isfinite(value) or end > filed or (start is not None and start > end):
            raise ValueError(f"Invalid SEC {concept} dates/value: {item!r}")
        if not accession:
            raise ValueError(f"Missing accession for SEC {concept}")
        if duration and not 70 <= (end - start).days + 1 <= 380:
            continue
        facts.append(
            {
                "start": start,
                "end": end,
                "filed": filed,
                "val": value,
                "accn": accession,
                "frame": bool(item.get("frame")),
            }
        )
    # A frame identifies a standalone context when otherwise identical contexts coexist.
    return sorted(facts, key=lambda fact: (fact["filed"], fact["accn"], fact["frame"]))


def _quarterly(known):
    """Use only the supplied known vintages; annual minus nine months produces Q4."""
    quarters = {}
    by_start = defaultdict(list)
    for fact in known.values():
        by_start[fact["start"]].append(fact)
        days = (fact["end"] - fact["start"]).days + 1
        # Retailers can have 16/17-week fourth quarters in a 52/53-week year.
        if 70 <= days <= 125:
            candidate = {**fact, "sources": {fact["accn"]}, "direct": True}
            previous = quarters.get(fact["end"])
            if previous is None or (fact["filed"], fact["accn"]) > (
                previous["filed"],
                previous["accn"],
            ):
                quarters[fact["end"]] = candidate
    for periods in by_start.values():
        periods.sort(key=lambda fact: fact["end"])
        for index, current in enumerate(periods):
            if current["end"] in quarters:
                continue  # Never replace a direct quarterly observation with a difference.
            earlier = [
                fact for fact in periods[:index] if 70 <= (current["end"] - fact["end"]).days <= 125
            ]
            if not earlier:
                continue
            prior = earlier[-1]
            quarters[current["end"]] = {
                **current,
                "start": prior["end"] + pd.Timedelta(days=1),
                "val": current["val"] - prior["val"],
                "sources": {current["accn"], prior["accn"]},
                "direct": False,
            }
    return quarters


def _earnings_for_concept(payload, ticker, concept) -> pd.DataFrame:
    """Extract new-quarter events from chronological SEC companyfacts vintages.

    Seasonal changes use the current filing's known current/prior-year quarterly
    earnings concept and assets exactly at the current quarter end. The denominator
    for standardization is the sample standard deviation of the last eight *prior*
    observable seasonal changes (minimum four); historical changes are frozen at
    their original availability. A missing asset, prior-year quarter or scale emits
    no event. Restatements cannot rejuvenate an already seen quarter.
    """
    income = _facts(payload, concept, True)
    assets = _facts(payload, "Assets", False)
    by_filing = defaultdict(list)
    for fact in income:
        by_filing[fact["filed"]].append(fact)
    known_income, known_assets = {}, {}
    asset_index = 0
    latest_end = None
    history, events = [], []
    coverage = {
        "income_facts": len(income),
        "asset_facts": len(assets),
        "new_quarters": 0,
        "missing_prior_year": 0,
        "missing_assets": 0,
        "insufficient_prior_changes": 0,
        "zero_scale": 0,
    }
    for filed, current_facts in sorted(by_filing.items()):
        for fact in current_facts:
            known_income[(fact["start"], fact["end"])] = fact
        while asset_index < len(assets) and assets[asset_index]["filed"] <= filed:
            fact = assets[asset_index]
            known_assets[fact["end"]] = fact
            asset_index += 1
        quarters = _quarterly(known_income)
        if not quarters:
            continue
        fiscal_end = max(quarters)
        if latest_end is not None and fiscal_end <= latest_end:
            continue
        latest_end = fiscal_end
        coverage["new_quarters"] += 1
        quarter = quarters[fiscal_end]
        prior_ends = [end for end in quarters if 350 <= (fiscal_end - end).days <= 380]
        if not prior_ends:
            coverage["missing_prior_year"] += 1
            continue
        prior_end = min(prior_ends, key=lambda end: abs((fiscal_end - end).days - 365))
        prior = quarters[prior_end]
        asset = known_assets.get(fiscal_end)
        if asset is None or asset["val"] <= 0:
            coverage["missing_assets"] += 1
            continue
        change = (quarter["val"] - prior["val"]) / asset["val"]
        if not math.isfinite(change):
            raise ValueError(f"Nonfinite seasonal earnings change for {ticker} at {filed.date()}")
        if len(history) >= 4:
            scale = float(np.std(history[-8:], ddof=1))
            if math.isfinite(scale) and scale > 0:
                events.append(
                    {
                        "ticker": ticker,
                        "filed_date": filed,
                        "fiscal_end": fiscal_end,
                        "surprise": float(np.clip(change / scale, -5, 5)),
                        "seasonal_change": change,
                        "source_concept": concept,
                        "source_accn": ";".join(
                            sorted(quarter["sources"] | prior["sources"] | {asset["accn"]})
                        ),
                    }
                )
            else:
                coverage["zero_scale"] += 1
        else:
            coverage["insufficient_prior_changes"] += 1
        history.append(change)
    result = pd.DataFrame(events, columns=EARNINGS_COLUMNS)
    for column in ("filed_date", "fiscal_end"):
        result[column] = pd.to_datetime(result[column])
    coverage["events"] = len(result)
    result.attrs["coverage"] = coverage
    return result


def earnings_from_companyfacts(payload, ticker) -> pd.DataFrame:
    """Merge coherent earnings definitions in their actual availability order.

    Some issuers use earnings available to common shareholders instead of
    NetIncomeLoss. Each concept gets its OWN seasonal differences and historical
    scale: never subtract two different definitions. Earliest usable disclosure
    wins; NetIncomeLoss is preferred only when both arrive on the same day.
    A later filing under a preferred tag cannot rewrite an older event.
    """
    concepts = ("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic")
    frames = [_earnings_for_concept(payload, ticker, concept) for concept in concepts]
    diagnostics = {concept: frame.attrs["coverage"] for concept, frame in zip(concepts, frames)}
    available = []
    for priority, frame in enumerate(frames):
        if not frame.empty:
            available.append(frame.assign(_priority=priority))
    if available:
        result = (
            pd.concat(available, ignore_index=True)
            .sort_values(["filed_date", "fiscal_end", "_priority"], ascending=[True, False, True])
            .drop_duplicates("filed_date")
        )
        previous_end = result["fiscal_end"].cummax().shift()
        result = result.loc[previous_end.isna() | (result["fiscal_end"] > previous_end)]
        result = result[EARNINGS_COLUMNS].reset_index(drop=True)
    else:
        result = frames[0].copy()
    result.attrs["coverage"] = {
        "events": len(result),
        "concepts": diagnostics,
        "selected_concepts": result["source_concept"].value_counts().to_dict(),
    }
    return result


def _fetch(url, user_agent):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        payload = json.loads(body)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot download {url}: {exc}. Check connectivity/provider limits; for SEC use "
            "a descriptive --sec-user-agent containing your institution and contact email. "
            "No synthetic data were substituted."
        ) from exc
    return body, payload


def _cached_json(raw_dir, key, url, user_agent, required_start=None, required_end=None):
    body_path = raw_dir / f"{key}.json"
    meta_path = raw_dir / f"{key}.meta.json"
    if body_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        body = body_path.read_bytes()
        if hashlib.sha256(body).hexdigest() != metadata.get("sha256"):
            raise ValueError(
                f"Raw cache checksum mismatch: {body_path}; remove this cache pair and download again"
            )
        sufficient = metadata.get("cache_version") == _CACHE_VERSION
        if required_start is not None:
            sufficient = sufficient and metadata.get("requested_start", "9999") <= required_start
            sufficient = (
                sufficient and metadata.get("coverage_end_exclusive", "0000") >= required_end
            )
        elif metadata.get("url") != url:
            sufficient = False
        if required_start is None and required_end is not None:
            fetched_day = datetime.fromisoformat(metadata["fetched_at"]).astimezone(_NY).date()
            sufficient = sufficient and fetched_day.isoformat() >= required_end
        if sufficient:
            return json.loads(body), metadata
    body, payload = _fetch(url, user_agent)
    metadata = {
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(),
        "sha256": hashlib.sha256(body).hexdigest(),
        "cache_version": _CACHE_VERSION,
    }
    if required_start is not None:
        metadata.update(
            requested_start=required_start,
            coverage_end_exclusive=datetime.now(_NY).date().isoformat(),
        )
    body_path.write_bytes(body)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return payload, metadata


def _yahoo_prices(payload, ticker, start, end):
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result") or len(chart["result"]) != 1:
        raise ValueError(f"Yahoo returned no valid chart for {ticker}: {chart.get('error')}")
    result = chart["result"][0]
    metadata = result.get("meta", {})
    if metadata.get("instrumentType") != "EQUITY" or metadata.get("currency") != "USD":
        raise ValueError(
            f"{ticker}: expected USD EQUITY, got {metadata.get('instrumentType')} / {metadata.get('currency')}"
        )
    returned_symbol = str(metadata.get("symbol", "")).upper()
    if returned_symbol != ticker.upper():
        raise ValueError(f"Yahoo symbol mismatch: requested {ticker}, returned {returned_symbol}")
    exchange_timezone = metadata.get("exchangeTimezoneName")
    if not exchange_timezone or not exchange_timezone.startswith("America/"):
        raise ValueError(f"{ticker}: unsupported/missing US exchange timezone: {exchange_timezone}")

    def session(timestamp):
        return (
            pd.Timestamp(timestamp, unit="s", tz="UTC")
            .tz_convert(exchange_timezone)
            .tz_localize(None)
            .normalize()
        )

    try:
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        frame = pd.DataFrame(
            {
                "date": [session(value) for value in timestamps],
                "ticker": ticker,
                "close": quote["close"],
                "adj_close": adjusted,
                "volume": quote["volume"],
            }
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"{ticker}: malformed Yahoo price arrays") from exc
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError(f"{ticker}: empty or duplicate Yahoo sessions")
    frame = frame.sort_values("date").reset_index(drop=True)
    splits = defaultdict(lambda: 1.0)
    dividends = defaultdict(float)
    events = result.get("events", {})
    for action in events.get("splits", {}).values():
        try:
            ratio = float(action["numerator"]) / float(action["denominator"])
            action_date = session(action["date"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{ticker}: malformed split action {action}") from exc
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"{ticker}: invalid split ratio {ratio}")
        splits[action_date] *= ratio
    for action in events.get("dividends", {}).values():
        try:
            amount = float(action["amount"])
            action_date = session(action["date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{ticker}: malformed dividend action {action}") from exc
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"{ticker}: invalid dividend amount {amount}")
        dividends[action_date] += amount
    frame["split"] = frame["date"].map(lambda date: splits.get(date, 1.0))
    frame["dividend"] = frame["date"].map(lambda date: dividends.get(date, 0.0))
    # Include actions AFTER the requested end and actions not represented by a price row.
    # Yahoo close/dividends are split adjusted; the split date itself is already on the new basis.
    ordered_splits = sorted(splits.items(), reverse=True)
    factors = np.ones(len(frame))
    factor, index = 1.0, 0
    for row_index in range(len(frame) - 1, -1, -1):
        date = frame.at[row_index, "date"]
        while index < len(ordered_splits) and ordered_splits[index][0] > date:
            factor *= ordered_splits[index][1]
            index += 1
        factors[row_index] = factor
    frame["close"] = pd.to_numeric(frame["close"], errors="raise") * factors
    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise") / factors
    frame["dividend"] *= factors
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] < end), PRICE_COLUMNS].copy()
    if frame.empty:
        raise ValueError(
            f"{ticker}: no prices in requested interval; check ticker history and dates"
        )
    sessions = set(frame["date"])
    missing_actions = [
        date
        for date in splits.keys() | dividends.keys()
        if start <= date < end and date not in sessions
    ]
    if missing_actions:
        raise ValueError(f"{ticker}: corporate actions without price sessions: {missing_actions}")
    _validate_prices(frame)
    return frame


def _validate_prices(frame):
    missing = set(PRICE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prices missing columns: {sorted(missing)}")
    if frame.empty or frame[["date", "ticker"]].isna().any().any():
        raise ValueError("Prices must contain nonmissing dates and tickers")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate ticker/date prices")
    for column in ("close", "adj_close", "volume", "dividend", "split"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Prices have missing/nonfinite {column}; no rows were filled or dropped"
            )
        if ((values <= 0) if column in {"close", "adj_close", "split"} else (values < 0)).any():
            raise ValueError(f"Prices have invalid {column}")


def download_dataset(
    universe_path, start, end, output_dir, sec_user_agent="STAT7008 university investment research"
) -> dict:
    """Download/cache real data for [start, end); fail explicitly on a ticker failure.

    Both source caches are reused while their snapshots cover the requested end.
    Extending beyond the SEC retrieval date refreshes filings as well as prices.
    Each price fetch runs through retrieval time to see all later splits.
    Current-day/incomplete session prices are not accepted as a requested endpoint.
    """
    start, end = _date(start, "start"), _date(end, "end")
    if start >= end:
        raise ValueError("Expected start < end (end is exclusive)")
    if end > pd.Timestamp(datetime.now(_NY).date()):
        raise ValueError(
            "end must be no later than today's New York date; current-session prices are incomplete"
        )
    if not sec_user_agent.strip():
        raise ValueError("A descriptive SEC user agent is required")
    universe = load_universe(universe_path)
    destination = Path(output_dir)
    raw_dir = destination / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prices, earnings, coverage, sources = [], [], {}, {}
    last_sec_request = 0.0
    for stock in universe.itertuples(index=False):
        try:
            params = urllib.parse.urlencode(
                {
                    "period1": int(start.tz_localize("UTC").timestamp()),
                    "period2": int(datetime.now(UTC).timestamp()),
                    "interval": "1d",
                    "events": "div,splits",
                    "includeAdjustedClose": "true",
                }
            )
            yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(stock.ticker)}?{params}"
            raw_prices, yahoo_source = _cached_json(
                raw_dir,
                f"yahoo_{stock.ticker}",
                yahoo_url,
                sec_user_agent,
                start.date().isoformat(),
                end.date().isoformat(),
            )
            price_frame = _yahoo_prices(raw_prices, stock.ticker, start, end)
            # Throttle even cache hits, so network requests are never closer than 250 ms.
            time.sleep(max(0.0, 0.25 - (time.monotonic() - last_sec_request)))
            last_sec_request = time.monotonic()
            sec_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{stock.cik}.json"
            raw_facts, sec_source = _cached_json(
                raw_dir,
                f"sec_{stock.cik}",
                sec_url,
                sec_user_agent,
                required_end=end.date().isoformat(),
            )
            if str(raw_facts.get("cik", "")).zfill(10) != stock.cik:
                raise ValueError(f"SEC CIK mismatch for {stock.ticker}")
            earnings_frame = earnings_from_companyfacts(raw_facts, stock.ticker)
            diagnostic = dict(earnings_frame.attrs["coverage"])
            # Retain pre-start events so downstream as-of signals have their real prior history.
            earnings_frame = earnings_frame.loc[earnings_frame["filed_date"] < end].copy()
            diagnostic.update(
                {
                    "events_before_end": len(earnings_frame),
                    "first_event": None
                    if earnings_frame.empty
                    else earnings_frame["filed_date"].min().date().isoformat(),
                    "last_event": None
                    if earnings_frame.empty
                    else earnings_frame["filed_date"].max().date().isoformat(),
                    "price_rows": len(price_frame),
                    "first_price": price_frame["date"].min().date().isoformat(),
                    "last_price": price_frame["date"].max().date().isoformat(),
                    "earnings_gap": len(earnings_frame) == 0,
                }
            )
            coverage[stock.ticker] = diagnostic
            sources[stock.ticker] = {"yahoo": yahoo_source, "sec": sec_source}
            prices.append(price_frame)
            earnings.append(earnings_frame)
        except Exception as exc:
            raise RuntimeError(
                f"Dataset download failed for {stock.ticker}: {exc}. "
                "Completed raw caches are retained; no incomplete normalized dataset was published."
            ) from exc
    all_prices = pd.concat(prices, ignore_index=True).sort_values(["date", "ticker"])
    all_earnings = pd.concat(earnings, ignore_index=True).sort_values(["filed_date", "ticker"])
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "requested_start": start.date().isoformat(),
        "requested_end_exclusive": end.date().isoformat(),
        "actual_price_start": all_prices["date"].min().date().isoformat(),
        "actual_price_end": all_prices["date"].max().date().isoformat(),
        "universe_type": "present-day survivor-selected US equities",
        "survivor_biased": True,
        "price_convention": "historical nominal close; split_t is new/old shares on date t",
        "price_return_formula": "close_t / close_previous * split_t - 1",
        "earnings_availability": "strictly after filed_date; fiscal_end is not availability",
        "earnings_scale": "sample std of up to 8 prior observed seasonal changes, min 4; clipped [-5,5]",
        "limitations": list(_LIMITATIONS),
        "coverage": coverage,
        "sources": sources,
        "files": {},
    }
    for name, frame in (
        ("prices.csv", all_prices),
        ("earnings.csv", all_earnings),
        ("universe.csv", universe),
    ):
        content = frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
        temporary = destination / f".{name}.tmp"
        temporary.write_bytes(content)
        metadata["files"][name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "rows": len(frame),
        }
    # Publish the manifest last; hashes detect interrupted updates rather than accepting mixed datasets.
    for name in metadata["files"]:
        (destination / f".{name}.tmp").replace(destination / name)
    manifest_tmp = destination / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    manifest_tmp.replace(destination / "manifest.json")
    return metadata


def _parse_dates(frame, columns, filename):
    for column in columns:
        if column not in frame:
            raise ValueError(f"{filename} missing {column}")
        values = pd.to_datetime(frame[column], errors="raise")
        if values.isna().any() or isinstance(values.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{filename}: {column} must be nonmissing timezone-naive dates")
        if not values.eq(values.dt.normalize()).all():
            raise ValueError(f"{filename}: {column} contains times instead of sessions")
        frame[column] = values


def load_dataset(data_dir) -> tuple:
    """Load normalized data, checking provenance hashes and numeric/date integrity."""
    directory = Path(data_dir)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No {manifest}; run the real-data download first")
    metadata = json.loads(manifest.read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported data schema; download a compatible dataset")
    for name in ("prices.csv", "earnings.csv", "universe.csv"):
        path = directory / name
        expected = metadata.get("files", {}).get(name, {}).get("sha256")
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Normalized dataset checksum mismatch: {path}; rerun download, do not mix snapshots"
            )
    universe = load_universe(directory / "universe.csv")
    prices = pd.read_csv(directory / "prices.csv", dtype={"ticker": str})
    earnings = pd.read_csv(directory / "earnings.csv", dtype={"ticker": str, "source_accn": str})
    _parse_dates(prices, ["date"], "prices.csv")
    _parse_dates(earnings, ["filed_date", "fiscal_end"], "earnings.csv")
    _validate_prices(prices)
    missing = set(EARNINGS_COLUMNS) - set(earnings.columns)
    if missing:
        raise ValueError(f"Earnings missing columns: {sorted(missing)}")
    if earnings[EARNINGS_COLUMNS].isna().any().any():
        raise ValueError("Earnings contain missing fields")
    if (
        earnings.duplicated(["ticker", "filed_date"]).any()
        or earnings.duplicated(["ticker", "fiscal_end"]).any()
    ):
        raise ValueError("Duplicate earnings event or reset fiscal-quarter freshness")
    if (earnings["fiscal_end"] > earnings["filed_date"]).any():
        raise ValueError("Earnings fiscal end exceeds filing date")
    for column in ("surprise", "seasonal_change"):
        earnings[column] = pd.to_numeric(earnings[column], errors="raise")
        if not np.isfinite(earnings[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Earnings have invalid {column}")
    if earnings["surprise"].abs().gt(5).any():
        raise ValueError("Earnings surprise outside clipped [-5,5] range")
    tickers = set(universe["ticker"])
    if set(prices["ticker"]) != tickers or not set(earnings["ticker"]).issubset(tickers):
        raise ValueError(
            "Dataset tickers do not match the universe; missing stocks cannot be silently dropped"
        )
    prices = prices[PRICE_COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)
    earnings = (
        earnings[EARNINGS_COLUMNS].sort_values(["filed_date", "ticker"]).reset_index(drop=True)
    )
    for ticker, group in earnings.groupby("ticker", sort=False):
        if not group["fiscal_end"].is_monotonic_increasing:
            raise ValueError(f"Earnings fiscal-quarter freshness runs backward for {ticker}")
    return prices, earnings, universe, metadata

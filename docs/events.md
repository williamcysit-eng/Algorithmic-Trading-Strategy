# Public-event source snapshots

`stock-events` is a separate information-availability tool. It captures original bytes and an append-only sequence of source revisions. It does not change trading scores, infer analyst consensus, parse earnings numbers automatically, or establish a profitable news signal.

Each record contains event identity, ticker, event kind, source URL, original publication metadata and its precision, the actual UTC collection timestamp, conservative availability, source SHA-256, a revision number, and a hash link to the previous ledger record. A corrected document appends a revision. Exact repeated imports are idempotent. As-of inspection selects the latest revision available at that timestamp, keeping older views stable.

Availability is the later of collection time and the original publication bound. An old release captured now never enters an old backtest. When a page supplies only a publication date, the bound is midnight at the start of the following day in the publication timezone; no publication clock time is invented. Consumers must use exact timezone-aware decision cutoffs before execution. A current webpage with a historical date is not a historical point-in-time archive.

## Capture or import

Install with `uv sync --frozen --extra dev`, then capture a real issuer release using its original HTTPS source and the publication date shown on the page:

```sh
uv run --frozen stock-events capture \
  --event-id AAPL-2025Q4-release --ticker AAPL --kind earnings_release \
  --source-url https://www.apple.com/newsroom/2025/10/apple-reports-fourth-quarter-results/ \
  --published 2025-10-30 --store data/events
```

If preserving already downloaded original bytes, replace `capture` with `import` and supply `--document PATH`. Import records the time of import rather than trusting the file timestamp as historical availability. `--fields PATH.json` optionally adds a strict JSON object with manually verified source metadata, including units/accounting basis when applicable. Such fields are assertions linked to the captured document; the tool does not certify extraction accuracy.

For SEC URLs, pass `--user-agent 'Project Name contact@example.org'` with your actual identification. The source capture rejects SEC requests without a contact address. Capture is limited to 10 MB and retains the full source response bytes. HTTP failure leaves no new ledger entry.

## Inspect information available at a decision

```sh
uv run --frozen stock-events as-of --store data/events --at 2025-11-05T21:00:00Z
```

A release first captured in 2026 is absent at that 2025 cutoff, even if its page says 2025. For a current cutoff use an exact UTC or offset timestamp. Only successfully observed records are returned. Absence means no captured record, not that no market event occurred.

Supported kinds are `earnings_release`, `earnings_schedule`, `dividend_schedule`, and `consensus_snapshot`. A consensus snapshot's collection time must precede an announcement before it can be used to measure that announcement's surprise. Earnings schedules and dividends need their own announcement times; realized future event dates cannot be retroactively substituted.

## Storage and integrity

The ignored `data/events/` directory contains `ledger.json` and content-addressed `objects/`. A writer lock prevents concurrent updates, and an atomic ledger replacement publishes a complete new revision sequence. Every read verifies source bytes, record hashes, revision identity/order, collection order, and availability. A failed write may leave an unreferenced source object, which does not enter as-of views.

The hash chain detects changed content and altered intermediate history; it is not an external signature and cannot by itself prove that a suffix was not removed. Archive ledger copies and publish external hashes when locking live decisions. Original sources may be revised or removed by their issuers. Retrieval does not establish when the original market first observed the information.

The ledger is deliberately separate from the legacy dataset schema. Feeding new features into trading requires a separately frozen experiment with genuine historical availability or prospective observations. The event foundation can be retained when an unrelated selection experiment is rolled back because it changes no investment decisions.

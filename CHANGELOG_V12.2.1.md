# V12.2.1 Changelog

## Bug fix

Fixed the V12.2 live-signal crash caused by mixing pandas UTC `Timestamp` values from REST warmup history with ISO timestamp strings emitted by the live `CandleStore` finalization callback.

The callback now normalizes all timestamps to UTC `pandas.Timestamp` before comparison, concatenation, and sorting.

Also replaced the deprecated `Series.view("int64")` timestamp conversion in warmup filtering with `astype("int64")`.

## Verification

- Python 3.13 compatible.
- V12.2 functional tests retained.
- Added a regression test for mixed timestamp normalization.
- Test result in build environment: **16 passed**.
- No order-submission code added or enabled.

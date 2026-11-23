CREATE TABLE observations (
    site TEXT NOT NULL COLLATE BINARY,
    timestamp_utc INTEGER NOT NULL,
    observed_ph_centi INTEGER NOT NULL CHECK (observed_ph_centi BETWEEN 0 AND 1400),
    baseline_ph_centi INTEGER NOT NULL CHECK (baseline_ph_centi BETWEEN 0 AND 1400),
    PRIMARY KEY (site, timestamp_utc)
) STRICT;

CREATE TABLE site_logs (
    id TEXT PRIMARY KEY,
    site TEXT NOT NULL COLLATE BINARY,
    timestamp_utc INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('维护', '校准')),
    note TEXT NOT NULL
) STRICT;

CREATE INDEX site_logs_site_time ON site_logs (site, timestamp_utc);

CREATE SCHEMA IF NOT EXISTS vees;

CREATE TABLE IF NOT EXISTS vees.events_clean (
    event_id            TEXT PRIMARY KEY,
    report_date         DATE,
    county              TEXT NOT NULL,
    sub_county          TEXT NOT NULL,
    ward                TEXT NOT NULL,
    locality            TEXT NOT NULL,
    species_affected    TEXT NOT NULL,
    number_at_risk      INTEGER NOT NULL DEFAULT 0,
    number_sick_bitten  NUMERIC NOT NULL DEFAULT 0,
    number_dead         INTEGER NOT NULL DEFAULT 0,
    disease_condition   TEXT NOT NULL,
    nature_of_diagnosis TEXT NOT NULL,
    humans_affected     NUMERIC NOT NULL DEFAULT 0,
    control_methods     TEXT[] NOT NULL DEFAULT '{}',
    longitude           DOUBLE PRECISION,
    latitude            DOUBLE PRECISION,
    number_sick         NUMERIC NOT NULL DEFAULT 0,
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_county      ON vees.events_clean (county);
CREATE INDEX IF NOT EXISTS idx_events_report_date ON vees.events_clean (report_date);
CREATE INDEX IF NOT EXISTS idx_events_disease     ON vees.events_clean (disease_condition);
CREATE INDEX IF NOT EXISTS idx_events_geo         ON vees.events_clean (latitude, longitude);

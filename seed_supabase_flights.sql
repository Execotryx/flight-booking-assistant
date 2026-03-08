-- Seed script for Supabase SQL Editor
-- Mirrors seed_supabase_flights.py behavior:
-- 1) Upsert flights on conflict (flight_id)
-- 2) Upsert city lookup rows on conflict (city_name)
--
-- Usage:
-- - Open Supabase dashboard -> SQL Editor
-- - Paste and run this script
--
-- Optional replace behavior (equivalent to --replace):
-- TRUNCATE TABLE public.flights RESTART IDENTITY;
-- TRUNCATE TABLE public.city_code_lookup RESTART IDENTITY;

BEGIN;

CREATE TABLE IF NOT EXISTS public.flights (
    flight_id text PRIMARY KEY,
    airline text NOT NULL,
    origin text NOT NULL,
    destination text NOT NULL,
    date date NOT NULL,
    depart_time time NOT NULL,
    arrive_time time NOT NULL,
    base_fare_usd integer NOT NULL CHECK (base_fare_usd >= 0),
    seats_left integer NOT NULL CHECK (seats_left >= 0)
);

CREATE TABLE IF NOT EXISTS public.city_code_lookup (
    city_name text PRIMARY KEY,
    iata_code text NOT NULL
);

CREATE TABLE IF NOT EXISTS public.bookings (
    booking_id text PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('confirmed', 'cancelled')),
    created_at timestamptz NOT NULL,
    cancelled_at timestamptz,
    passenger_name text NOT NULL,
    flight_id text NOT NULL REFERENCES public.flights(flight_id),
    airline text NOT NULL,
    origin text NOT NULL,
    destination text NOT NULL,
    date date NOT NULL,
    depart_time time NOT NULL,
    arrive_time time NOT NULL,
    cabin_class text NOT NULL,
    paid_fare_usd integer NOT NULL CHECK (paid_fare_usd >= 0)
);

INSERT INTO public.flights (
    flight_id,
    airline,
    origin,
    destination,
    date,
    depart_time,
    arrive_time,
    base_fare_usd,
    seats_left
)
VALUES
    ('FL-1001', 'SkyJet', 'NYC', 'LON', '2026-03-20', '09:10', '21:45', 640, 4),
    ('FL-1002', 'AeroWays', 'NYC', 'LON', '2026-03-20', '18:35', '07:05', 560, 7),
    ('FL-2001', 'PacificAir', 'LAX', 'NRT', '2026-03-22', '11:40', '16:10', 790, 5),
    ('FL-3001', 'EuroConnect', 'LON', 'CDG', '2026-03-18', '14:15', '16:20', 120, 9)
ON CONFLICT (flight_id)
DO UPDATE SET
    airline = EXCLUDED.airline,
    origin = EXCLUDED.origin,
    destination = EXCLUDED.destination,
    date = EXCLUDED.date,
    depart_time = EXCLUDED.depart_time,
    arrive_time = EXCLUDED.arrive_time,
    base_fare_usd = EXCLUDED.base_fare_usd,
    seats_left = EXCLUDED.seats_left;

INSERT INTO public.city_code_lookup (city_name, iata_code)
VALUES
    ('new york', 'NYC'),
    ('nyc', 'NYC'),
    ('london', 'LON'),
    ('lon', 'LON'),
    ('paris', 'CDG'),
    ('cdg', 'CDG'),
    ('los angeles', 'LAX'),
    ('lax', 'LAX'),
    ('tokyo', 'NRT'),
    ('nrt', 'NRT')
ON CONFLICT (city_name)
DO UPDATE SET
    iata_code = EXCLUDED.iata_code;

COMMIT;

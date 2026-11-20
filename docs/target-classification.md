# Target classification

Every entry in the aircraft feed carries up to two independent answers to
"what is this?". The first is the **label** from the target's ICAO hex. The
second is the **motion baseline**, read from how the target moves. Both use one
vocabulary:

| Class | What it covers | Vancouver examples |
| --- | --- | --- |
| `jet` | Turbofan and turbojet fixed wing, any size | YVR airliners (737, A320, 777), bizjets |
| `turboprop` | Turbine-driven propeller fixed wing | Dash 8 and Saab 340 regionals, King Air, turbine Otters and Caravans |
| `piston` | Piston and electric prop-driven fixed wing, ultralights | Boundary Bay and Pitt Meadows trainers (C172, PA-28), Beaver floatplanes, Harbour Air's electric Beaver |
| `helicopter` | Any rotary wing | Helijet, air ambulance, police |
| `drone` | UAVs | Small multirotors and fixed-wing UAVs |
| `other` | Gliders, balloons and airships, gyrocopters, tiltrotors, ground vehicles, towers. Label only: the baseline never predicts it and the scorecard does not record it | Gliders, balloons, airport vehicles |

The classes are bins of what produces the Doppler signature, not of airframe
shape. The end goal is Doppler and micro-Doppler classification, so the labels
follow the source of the return: a jet's compressor and turbine blades, a
propeller's blades, a rotor's blades, or only a small body. A Dash 8 and an
A320 look alike to a motion model at cruise, but their micro-Doppler differs,
which is why they are different classes. Jet and turboprop motion envelopes
overlap on purpose. Separating them is expected to need Doppler, and the
scorecard is meant to show it.

## 1. The label: hex lookup

`services/target_identity.py`, backed by `services/aircraft_db.py`.

- **Aircraft registry.** This is the tar1090-db export of Mictronics' readsb
  database, the same data tar1090 labels aircraft with. It has ~620k rows of
  hex, registration, type designator, flags, description and operator. A type
  table maps each designator to its ICAO class (`L2J`, `H1T`...) and wake
  category. `class_from_icao` reduces that to a class above.
- **ADS-B emitter category** (`A7`, `B1`–`B7`, `C1`–`C3`). This is the
  fallback for hexes the registry doesn't have. `A7` is a helicopter, `B6` a
  drone, `B4` (ultralight) piston, and `B1`–`B3`, `B7` and `C1`–`C3` other.
  `A1`–`A6` are deliberately unmapped: they are weight classes, and a "small"
  or "large" aircraft can be a jet, a turboprop or a piston twin, so the
  category cannot make the one distinction the bins turn on. Nodes don't send it today. The
  adsb.lol cache does, and any ADS-B record that carries `category` is read.

The registry is converted once into a SQLite file,
`backend/data/aircraft_db.sqlite` (or `AIRCRAFT_DB_PATH`). Each lookup is one
indexed read behind an LRU cache, so 620k rows never sit in the server's
memory. The file comes from one of two places:

- the `aircraft_db_refresh` task, where `AIRCRAFT_DB_ENABLED=1`, which builds
  it at boot and rebuilds it weekly. Staging sets the flag. Production doesn't
  yet.
- `python scripts/fetch_aircraft_db.py`, by hand, anywhere.

With no file, nothing is labelled and the feed is unchanged. A file that is
built or replaced while the server runs is picked up on the next lookup.

The feed entry gets `identity`:

```json
{"source": "aircraft_db", "class": "turboprop", "military": false,
 "registration": "C-FDHD", "type_code": "DH8D", "type_name": "DHC-8-400 Dash 8Q",
 "icao_class": "L2T", "wtc": "M", "operator": "Porter", "blades": 6}
```

The registry outranks any inference, so where the label has a class, the
legacy `target_class` (`aircraft` / `drone`) follows it.

### Blade counts

`BLADES` in `target_identity.py` maps a type designator to its propeller or
main-rotor blade count, and the identity carries `blades` when the type is in
it. Blade count sets the spacing of the micro-Doppler lines (blade-passing
frequency is blades times rotation rate), so it is the one airframe fact a
Doppler classifier wants beyond the class. The values are for the common
variant of each type, from general knowledge, and should be verified.

| Type | Blades | Type | Blades | Type | Blades |
| --- | --- | --- | --- | --- | --- |
| DH8D | 6 | DH8A / DH8B / DH8C | 4 | AT72 / AT76 / AT45 / AT75 | 6 |
| SF34 | 4 | DHC6 | 3 | C172 / C152 / PA28 | 2 |
| R44 / R22 | 2 | AS50 | 3 | EC35 / S76 / B412 | 4 |
| B06 | 2 | EC30 | 3 | A139 | 5 |

Three cases are never looked up:

- synthetic-fleet hexes, which are random and would match some real
  aircraft;
- non-ICAO `~` addresses;
- solver-minted `mn…`/`pr…` hexes.

## 2. The motion baseline

`services/motion_classifier.py`. It uses no identity, no signature and no
learned weights.

Features come from the target's trail in `state.track_histories`:

- median speed and 90th-percentile speed;
- altitude;
- median turn rate and median climb rate;
- the fraction of time spent hovering.

Every rate is taken over chords at least 10 s long. A multinode solve is only
good to a few hundred metres, so differencing consecutive 1 Hz points would
measure the noise.

Each class has a trapezoidal envelope per feature. The flat top is how the
class typically flies; the tapers show how far it can stretch. A class scores
its prior × the product of its memberships. A class that cannot hover (jet, turboprop, piston) is
penalised. The scores are normalised into a distribution.

A feature the target didn't yield is left out rather than guessed. The verdict
is `unknown` unless the best class reaches 0.35 and beats its own prior by 25%.
Without that rule the prior alone would answer for a target that has shown
nothing.

Only `multinode_solve` and `adsb_single_node` positions are measured
positions, so only their trails are read as motion. A single-node arc's drawn
point is the arc midpoint, so its trail records the delay changing, not the
target moving. Those entries use their **Doppler floor** instead: bistatic
range rate is at most twice the target's speed, so `|v| ≥ λ·|f_d| / 2`. At
195 MHz, −150 Hz means at least 115 m/s, which rules out every slow class.

The feed entry gets `motion_class`:

```json
{"class": "jet", "confidence": 0.98,
 "scores": {"jet": 0.98, "...": 0.0},
 "features": {"span_s": 60.0, "n_chords": 6, "speed_ms": 231.2, "alt_m": 11000.0}}
```

The baseline does not set `target_class`. It is published beside it until the
scorecard shows it has earned that.

## 3. Scoring the baseline

Wherever a target has both a labelled class and a motion verdict, the pair is
a labelled example. `GET /api/admin/classification/scorecard` reports, over the
last 6 h with one vote per transponder (its latest pair):

- **accuracy**: over the targets the baseline committed on;
- **coverage**: how many targets it committed on at all;
- per-class recall and precision;
- the full confusion matrix, `unknown` included;
- `by_altitude`: the same confusion matrix, with targets, decided and
  accuracy, for each band of the target's altitude (`<1km`, `1-3km`,
  `3-6km`, `>6km`, `unknown`). It shows whether motion separates a jet from a
  turboprop only at cruise;
- whether the registry is loaded, and how old it is.

That number is the one a learned classifier has to beat.

## 4. Real data: from a node's detections to a dataset

Nothing here needs a code change once a node streams detections. The flow:

1. **Node frames** arrive with parallel `delay`, `doppler` and `snr` arrays and
   go through `PassiveRadarPipeline.process_frame`, like simulated ones.
2. **Tracker:** builds tracks per node; multinode solves where nodes overlap.
3. **Comb features:** after each frame, every track whose newest detection came
   from that frame is checked for extra detections at its delay
   (`services/comb_features.py`). The tracker's own association picks the body
   line. Detections it gave other tracks are excluded. The summary rides on the
   feed entry as `comb`.
4. **Motion classifier + hex label:** `classify_feed` annotates every feed
   entry with `motion_class` and, where a transponder names it, `identity`.
5. **Archive:** `services/classification_archive.py` folds each target into
   60 s windows and writes them hourly to Parquet at
   `backend/data/classification_archive/YYYY/MM/DD/HH.parquet`. That is the
   `backend-data` volume, `/app/backend/data` inside the container.
6. **Report / export:** `scripts/classification_report.py` prints the
   scorecard, comb tables and jet-vs-turboprop table, and `--export` writes a
   training set with a `group` column (the hex) for splitting by aircraft.

Environment:

| Variable | Default | Effect |
| --- | --- | --- |
| `AIRCRAFT_DB_ENABLED` | off (on in staging and production compose) | Downloads the registry. Without it real traffic gets no hex label |
| `CLASSIFICATION_ARCHIVE_ENABLED` | on unless exactly `0` | Writes the archive |

Running the report on the server:

```bash
docker compose exec backend python scripts/classification_report.py \
  data/classification_archive --world real --label-source hex \
  --export data/classification-real.parquet
```

`GET /api/admin/classification/scorecard` also shows the archive's buffered
and written row counts and its last flush.

What a node needs before its data is useful:

- **ADS-B enabled.** Detections carry a hex only when the node's ADS-B truth
  source is on. Without it the node's targets are classified but unlabelled.
- **A Doppler window of about ±1000 Hz** at ~500 MHz. At that carrier an
  airliner's Doppler and a rotor's comb both reach several hundred Hz, and a
  ±300 Hz window clips them.
- **Overlap with other nodes,** for multinode tracks. A lone node mostly
  yields the single-node Doppler speed floor, which is why the baseline answers
  `unknown` for most windows.

Known issue: in simulation, rotor comb lines produce spurious confirmed tracks.
See `docs/design-notes/2026-11-17-comb-lines-vs-tracker.md`.

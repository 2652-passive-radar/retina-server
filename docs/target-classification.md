# Target classification

Every entry in the aircraft feed carries up to two independent answers to
"what is this?". The first is the **label** from the target's ICAO hex. The
second is the **motion baseline**, read from how the target moves. Both use one
vocabulary:

| Class | What it covers |
| --- | --- |
| `fast_fixed_wing` | Jets of any size, and turboprops of medium wake or larger (ATR, Dash 8) |
| `light_fixed_wing` | Piston and single turboprop (C172, Caravan), ultralights |
| `rotorcraft` | Helicopters, gyrocopters, tiltrotors |
| `glider` | Gliders |
| `balloon` | Balloons and airships |
| `drone` | UAVs |
| `surface` | Ground vehicles and towers. Label only: the baseline never predicts it |

The classes are split by flight envelope, not by airframe. Motion can't tell a
bizjet from an airliner, so the vocabulary doesn't ask it to.

## 1. The label: hex lookup

`services/target_identity.py`, backed by `services/aircraft_db.py`.

- **Aircraft registry.** This is the tar1090-db export of Mictronics' readsb
  database, the same data tar1090 labels aircraft with. It has ~620k rows of
  hex, registration, type designator, flags, description and operator. A type
  table maps each designator to its ICAO class (`L2J`, `H1T`...) and wake
  category. `class_from_icao` reduces that to a class above.
- **ADS-B emitter category** (`A1`–`A7`, `B1`–`B7`, `C1`–`C3`). This is the
  fallback for hexes the registry doesn't have. Nodes don't send it today. The
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
{"source": "aircraft_db", "class": "fast_fixed_wing", "military": false,
 "registration": "EI-DCL", "type_code": "B738", "type_name": "BOEING 737-800",
 "icao_class": "L2J", "wtc": "M", "operator": "Ryanair"}
```

The registry outranks any inference, so where the label has a class, the
legacy `target_class` (`aircraft` / `drone`) follows it.

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
its prior × the product of its memberships. A fixed-wing class that hovers is
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
{"class": "fast_fixed_wing", "confidence": 0.98,
 "scores": {"fast_fixed_wing": 0.98, "...": 0.0},
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
- whether the registry is loaded, and how old it is.

That number is the one a learned classifier has to beat.

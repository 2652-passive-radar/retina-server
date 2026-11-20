# Comb lines vs the tracker (2026-11-17)

**Question:** do the simulator's rotor and propeller comb lines make the tracker
confirm tracks that are not aircraft?

**Answer:** in simulation, yes, heavily.

Commands, from `backend/`, with `RETINA_ENV=dev`. The centre is the default
example receiver area:

```bash
python scripts/classification_bench.py --tower-set yvr3 --nodes 3 --layout colocated --duration-s 1800 --micro-doppler
python scripts/classification_bench.py --tower-set yvr3 --nodes 3 --layout colocated --duration-s 1800 --no-micro-doppler
```

| | micro-Doppler on | micro-Doppler off |
| --- | --- | --- |
| Confirmed tracker tracks (all nodes) | 207 | 71 |
| Spurious: never inside any target's 1 µs / 10 Hz body gate | 114 | 1 |
| ...of those, at a target's delay but off its Doppler (rotor lines) | 111 | 0 |
| Spurious confirmed tracks per hour | 228 | 2 |
| Targets in the run | 19 | 18 |

Each comb line is a detection at the target's delay, at a Doppler offset that
repeats frame after frame. To the tracker that looks like a second target
flying alongside, so it confirms it. With micro-Doppler off the tracker is
essentially clean.

## How far this transfers to real nodes

It depends on whether blah2's CFAR reports comb lines as separate detections,
which is still unknown. The simulator assumes it does. If real nodes do the
same, every helicopter or propeller aircraft would bring a cluster of ghost
tracks with it.

## Options (none implemented)

- **Suppress before tracking.** Drop detections at an existing track's delay
  whose Doppler offset fits that track's comb spacing, and keep them for the
  comb features instead.
- **Gate in the tracker.** Refuse to start a track inside an existing
  confirmed track's delay gate.
- **Mark, don't drop.** Tag the extra tracks as rotor lines so the map and
  the solver ignore them.

Check this against a real recording of a helicopter or turboprop before
choosing. The comb features already exclude detections the tracker gave other
tracks, so ghost tracks would also starve the features of the very lines
they need.

# Classification bench, first run (2026-11-16)

First run of `backend/scripts/classification_bench.py`: the simulator's world through the
per-node tracker, track-level association and multinode solve, then one dataset row per target
per 60 s window and a scorecard of the motion baseline against simulator truth.

Command (from `backend/`, defaults throughout, wall time 92 s):

```
RETINA_ENV=dev python scripts/classification_bench.py --out data/classification-first-run.parquet
```

The Parquet file is not committed (`backend/data/*.parquet` is gitignored).

## Caveats

- **Small sample.** 14 targets in 15 simulated minutes (one turboprop, one drone), so per-class
  numbers are anecdotes, not rates. Scale `--duration-s`, `--nodes` and `--max-aircraft`, and
  repeat over `--seed`, before quoting anything.
- **Radar SNR model.** `--snr-model radar` (the default here): bistatic radar equation with a
  per-airframe RCS, SNR capped at 60 dB. Its reference constants are placeholders. Under the
  simulator's older `legacy` model, which caps SNR at 25 dB, rotor lines are almost never above
  the detection floor (about 0.0-0.06 lines per frame), so a comb would show nothing.
- **Interim micro-Doppler line strength.** Rotor amplitudes in `retina_simulation.micro_doppler`
  are calibrated so the strongest line sits 15-17 dB under the body. They are not fitted to a
  real node recording, so line counts and relative SNR are the simulator's, not a measurement.
- **Perfect association.** Each frame's body detection is found by matching the target's true
  delay and Doppler. A real receiver has only its tracker's association, so comb features here are
  an upper bound on what a deployment sees.
- **Geometry.** One illuminator for every node (`--tower-set chan`: CHAN-DT, 521 MHz). The tower
  altitude (2135 ft) assumes the tower table's columns are site elevation plus antenna height
  above ground. Receivers are an example Vancouver west-side area, 3-15 km from the centre, not a
  surveyed site. `--tower-set yvr3` (CHAN-DT, CIVI-DT-2, CHNU-DT) is the planned three-node setup;
  CIVI-DT-2 and CHNU-DT share Mt Seymour's main site, so it has two transmitter sites and differs
  from `chan` more in frequency than in geometry. It was not run here.
- **Motion baseline inputs.** A window is read as a multinode trail only with at least 3 matched
  solves spanning 10 s; otherwise from the largest Doppler speed floor in the window. Most rows are
  the latter (no altitude), so the altitude bands are mostly `unknown`.
- **Simulator source.** `libs/retina-simulation` points at 7e56c22 on an unmerged PR branch; this
  report must be regenerated if that PR changes the simulator.

## Reading it

- The baseline commits on 46% of windows and is 78% accurate on those, but it only ever says
  `jet` for anything fast: all 8 decided turboprop windows were called jet (recall 0), the
  confusion this work exists to break. The comb separates them here: no jet window had 2 or more
  lines, half of the turboprop windows did, with median spacing 99.7 Hz against 102 Hz expected.
- Piston spacing (91 Hz against 90) and helicopter (13.2 against 13.6) are recovered too. The
  helicopter comb is sparse (3% of frames) and the baseline calls every helicopter window unknown.
- Jet rows with a "comb" are single stray detections at the body's delay (1 line, spacing
  meaningless); `n_lines >= 2` is the useful gate. The two drone rows give 13 Hz against 180
  expected, too few to read.

## Full printed report

```
classification bench
command: python scripts/classification_bench.py --out data/classification-first-run.parquet
scene: 4 nodes 3-15 km from 49.26,-123.25 (Vancouver west side, example receiver area), 900s, seed 42, doppler +/-1000 Hz, cpi=0.5s, micro-doppler=on, snr-model=radar, blind, mix=jet:0.45,turboprop:0.15,piston:0.2,helicopter:0.1,drone:0.1
  node cls-00: 49.3140,-123.2500  tower 49.345000,-122.972778 @ 2135 ft  fc=521 MHz
  node cls-01: 49.2600,-123.0846  tower 49.345000,-122.972778 @ 2135 ft  fc=521 MHz
  node cls-02: 49.2240,-123.2500  tower 49.345000,-122.972778 @ 2135 ft  fc=521 MHz
  node cls-03: 49.2600,-123.4567  tower 49.345000,-122.972778 @ 2135 ft  fc=521 MHz
towers: --tower-set chan
ran 1801 frames, 299 solves matched to a target in 92s wall

rows and targets per class (truth airframe)
  class         targets   rows    rows from multinode solves
  jet                 7     46                             6
  turboprop           1     10                             2
  piston              2     18                             2
  helicopter          3     34                             0
  drone               1      2                             2
  total: 110 rows, 14 targets

motion baseline vs simulator truth, per row (window) [110 rows]
  targets=110 decided=51 accuracy=0.784 coverage=0.464
  truth \ predicted          jet  turboprop     piston helicopter      drone    unknown
  jet                        40          0          0          0          0          6
  turboprop                   8          0          0          0          0          2
  piston                      1          1          0          0          0         16
  helicopter                  0          0          0          0          0         34
  drone                       0          0          1          0          0          1
  per class: jet: recall=1.0 precision=0.816  turboprop: recall=0.0 precision=0.0  piston: recall=0.0 precision=0.0  helicopter: recall=None precision=None  drone: recall=0.0 precision=None
  by altitude band (targets / decided / accuracy): <1km: 0/0/None; 1-3km: 2/1/0.0; 3-6km: 7/4/0.25; >6km: 3/2/1.0; unknown: 98/44/0.841

motion baseline vs simulator truth, one vote per aircraft
  targets=14 decided=9 accuracy=0.778 coverage=0.643
  truth \ predicted          jet  turboprop     piston helicopter      drone    unknown
  jet                         7          0          0          0          0          0
  turboprop                   1          0          0          0          0          0
  piston                      0          0          0          0          0          2
  helicopter                  0          0          0          0          0          3
  drone                       0          0          1          0          0          0
  per class: jet: recall=1.0 precision=0.875  turboprop: recall=0.0 precision=None  piston: recall=None precision=0.0  helicopter: recall=None precision=None  drone: recall=0.0 precision=None
  by altitude band (targets / decided / accuracy): <1km: 0/0/None; 1-3km: 1/0/None; 3-6km: 5/5/0.6; >6km: 2/2/1.0; unknown: 6/2/1.0

comb features per class (rows; medians over rows that showed a comb)
  class         rows comb rows  frames w/ comb  n_lines  spacing Hz  expected  spread  symmetry  rel SNR
  jet             46        20            0.00      1.0        92.3         -     0.0      0.00    -11.8
  turboprop       10         7            0.80      4.0        99.7     102.0     0.4      0.67    -15.0
  piston          18         8            0.00      1.5        91.4      90.0     0.1      0.00    -10.7
  helicopter      34        17            0.03      7.0        13.2      13.6    10.3      0.29    -16.7
  drone            2         2            0.17      6.5        13.2     180.0     6.1      0.00      3.9

jet vs turboprop: where motion cannot split them, does the comb?
  jet        rows=  46  comb of >=2 lines in 0/46 rows; baseline calls it: {'jet': 40, 'unknown': 6}
  turboprop  rows=  10  comb of >=2 lines in 5/10 rows; baseline calls it: {'jet': 8, 'unknown': 2}

wrote 110 rows to data/classification-first-run.parquet
```

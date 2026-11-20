#!/usr/bin/env python3
"""Score the motion baseline on a classification dataset, and export it for training.

Reads the Parquet the live archive writes (services.classification_archive,
``data/classification_archive/YYYY/MM/DD/HH.parquet``) and the bench writes
(scripts/classification_bench.py --out), one file or a directory tree of them,
both in the columns services.classification_columns defines.  It prints:

  * the motion baseline's scorecard against the label, per window and as one
    vote per target: confusion matrix, per-class recall and precision, by
    altitude band, and by where the label came from.  The arithmetic is the
    live scorecard's own (target_classification.build_card);
  * the comb features per class and the jet-versus-turboprop table, laid out as
    in the bench's report.

``--export`` writes the filtered, labelled rows as a training set with a
``group`` column (the target's hex) so a split can keep every window of one
aircraft on one side: a target's windows are not independent.

A bench file has no ``world`` or ``label_source`` column, because every row is
simulated and labelled by the simulator; it reads as world "sim", label source
"sim_truth", and its ``object_id`` is its hex.  The default ``--world real``
therefore selects archive rows only: pass ``--world sim`` or ``all`` for a bench
export.  ``--since`` and ``--until`` compare against ``window_start_s``, which is
epoch seconds in the archive and simulated seconds in a bench file.

Usage:  python scripts/classification_report.py data/classification_archive
        python scripts/classification_report.py data/classification_archive --since 2026-11-01 \
            --label-source hex --export data/train.parquet
"""

import argparse
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from services import classification_columns as cols  # noqa: E402
from services import motion_classifier, target_classification  # noqa: E402
from services.target_identity import TARGET_CLASSES  # noqa: E402

# Expected comb spacing by class from retina_simulation.micro_doppler.AIRFRAMES: the
# strongest rotor's blades x rev/s.  Printed beside the measured medians.  Real
# aircraft differ; it is the simulator's number and a reference line, not a truth.
EXPECTED_SPACING_HZ = {"jet": None, "turboprop": 102.0, "piston": 90.0, "helicopter": 13.6, "drone": 180.0}

EXPORT_COLUMNS = {**cols.ARCHIVE_COLUMNS, "group": pa.string()}


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def _parquet_files(path: Path) -> list[Path]:
    if path.is_dir():
        # Dot-files are a flush's temp files, still being written.
        return sorted(p for p in path.rglob("*.parquet") if not p.name.startswith("."))
    return [path] if path.is_file() else []


def load_rows(paths: list[str]) -> tuple[list[dict], list[str]]:
    """Every row of every Parquet under ``paths``, normalised to the archive's names.

    Returns ``(rows, problems)``: a file that cannot be read is reported and
    skipped rather than ending the run, since an archive tree is written by a
    live server and can hold a half-finished file.
    """
    rows, problems = [], []
    for raw in paths:
        files = _parquet_files(Path(raw))
        if not files:
            problems.append(f"{raw}: no Parquet files found")
        for f in files:
            try:
                table = pq.read_table(f)
            except Exception as exc:
                problems.append(f"{f}: unreadable ({type(exc).__name__}: {exc})")
                continue
            bench = "world" not in table.column_names
            for row in table.to_pylist():
                if cols.GROUP_COLUMN not in row:
                    row[cols.GROUP_COLUMN] = row.get("object_id")
                if bench:
                    row["world"] = "sim"
                    row["label_source"] = "sim_truth"
                rows.append(row)
    return rows, problems


def _parse_time(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        t = datetime.fromisoformat(text)
        return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


def select(rows, since=None, until=None, world="real", label_source="any") -> list[dict]:
    out = []
    for r in rows:
        t = r.get("window_start_s")
        if since is not None and (t is None or t < since):
            continue
        if until is not None and (t is None or t >= until):
            continue
        if world != "all" and r.get("world") != world:
            continue
        if label_source != "any" and r.get("label_source") != label_source:
            continue
        out.append(r)
    return out


def labelled(rows) -> list[dict]:
    return [r for r in rows if r.get(cols.LABEL_COLUMN) in TARGET_CLASSES]


def vote_per_target(rows, group=cols.GROUP_COLUMN):
    """One (truth, verdict, alt) per target: its most common decided verdict, unknown on a tie."""
    by = defaultdict(list)
    for r in rows:
        by[r.get(group)].append(r)
    out = []
    for rs in by.values():
        counts = Counter(
            r["baseline_class"] for r in rs if r["baseline_class"] not in (None, motion_classifier.UNKNOWN)
        )
        top = counts.most_common(2)
        verdict = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else motion_classifier.UNKNOWN
        out.append((rs[0][cols.LABEL_COLUMN], verdict, median([r["alt_m"] for r in rs])))
    return out


def window_triples(rows):
    return [(r[cols.LABEL_COLUMN], r["baseline_class"] or motion_classifier.UNKNOWN, r["alt_m"]) for r in rows]


def print_card(title, card):
    print(f"\n{title}")
    print(
        f"  targets={card['targets']} decided={card['decided']} accuracy={card['accuracy']} coverage={card['coverage']}"
    )
    labels = list(TARGET_CLASSES) + [motion_classifier.UNKNOWN]
    print("  truth \\ predicted  " + "".join(f"{p:>11}" for p in labels))
    for truth in TARGET_CLASSES:
        print(f"  {truth:<18}" + "".join(f"{card['confusion'][truth][p]:>11}" for p in labels))
    print(
        "  per class: "
        + "  ".join(f"{c}: recall={v['recall']} precision={v['precision']}" for c, v in card["per_class"].items())
    )
    print(
        "  by altitude band (targets / decided / accuracy): "
        + "; ".join(f"{b}: {v['targets']}/{v['decided']}/{v['accuracy']}" for b, v in card["by_altitude"].items())
    )


def print_comb_tables(rows, label_col=cols.LABEL_COLUMN):
    print("\ncomb features per class (rows; medians over rows that showed a comb)")
    print(
        f"  {'class':<12}{'rows':>6}{'comb rows':>10}{'frames w/ comb':>16}{'n_lines':>9}{'spacing Hz':>12}"
        f"{'expected':>10}{'spread':>8}{'symmetry':>10}{'rel SNR':>9}"
    )
    for c in TARGET_CLASSES:
        rs = [r for r in rows if r[label_col] == c]
        comb = [r for r in rs if r["comb_median_n_lines"] is not None]
        print(
            f"  {c:<12}{len(rs):>6}{len(comb):>10}{fmt(median([r['comb_frac_frames'] for r in rs]), 2):>16}"
            f"{fmt(median([r['comb_median_n_lines'] for r in comb])):>9}"
            f"{fmt(median([r['comb_median_spacing_hz'] for r in comb])):>12}"
            f"{fmt(EXPECTED_SPACING_HZ[c]):>10}"
            f"{fmt(median([r['comb_spacing_spread_hz'] for r in comb])):>8}"
            f"{fmt(median([r['comb_median_symmetry'] for r in comb]), 2):>10}"
            f"{fmt(median([r['comb_median_rel_snr_db'] for r in comb])):>9}"
        )
    print("\njet vs turboprop: where motion cannot split them, does the comb?")
    for c in ("jet", "turboprop"):
        rs = [r for r in rows if r[label_col] == c]
        has = sum(1 for r in rs if (r["comb_median_n_lines"] or 0) >= 2)
        print(
            f"  {c:<10} rows={len(rs):>4}  comb of >=2 lines in {has}/{len(rs)} rows; "
            f"baseline calls it: {dict(Counter(r['baseline_class'] for r in rs))}"
        )


def export(rows, path) -> int:
    """Write the labelled ``rows`` as a training set, one ``group`` per target; returns the row count."""
    out = [{**{c: r.get(c) for c in cols.ARCHIVE_COLUMNS}, "group": r.get(cols.GROUP_COLUMN)} for r in labelled(rows)]
    if out:
        cols.write_parquet(out, path, EXPORT_COLUMNS)
    return len(out)


def report(rows: list[dict]) -> None:
    lab = labelled(rows)
    print(
        f"{len(rows)} rows, {len(lab)} labelled with a class the baseline predicts, {len({r['hex'] for r in lab})} targets"
    )
    if not lab:
        print("nothing labelled to score: the archive holds rows, but no target has a hex or simulator label yet")
        return
    print_card(
        f"motion baseline vs label, per row (window) [{len(lab)} rows]",
        target_classification.build_card(window_triples(lab)),
    )
    print_card(
        "motion baseline vs label, one vote per target (hex)", target_classification.build_card(vote_per_target(lab))
    )
    for source in cols.LABEL_SOURCES:
        sub = [r for r in lab if r.get("label_source") == source]
        if sub:
            print_card(
                f"label source {source}: one vote per target [{len(sub)} rows]",
                target_classification.build_card(vote_per_target(sub)),
            )
    print_comb_tables(lab)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="Parquet files or directories of them (the archive tree, bench exports)")
    p.add_argument("--since", type=_parse_time, help="window_start_s >= this: epoch seconds or ISO date/time (UTC)")
    p.add_argument("--until", type=_parse_time, help="window_start_s < this")
    p.add_argument("--world", choices=("real", "sim", "all"), default="real")
    p.add_argument("--label-source", choices=("hex", "sim_truth", "any"), default="any")
    p.add_argument("--export", metavar="PATH.parquet", help="write the filtered, labelled rows as a training set")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows, problems = load_rows(args.paths)
    for line in problems:
        print(f"warning: {line}", file=sys.stderr)
    print("classification report")
    print(f"inputs: {', '.join(args.paths)}")
    if not rows:
        print("no rows: the archive is empty or missing (nothing has been flushed yet, or the path is wrong)")
        return 0
    chosen = select(rows, args.since, args.until, args.world, args.label_source)
    print(
        f"filters: world={args.world} label-source={args.label_source} "
        f"since={args.since} until={args.until}  ->  {len(chosen)} of {len(rows)} rows"
    )
    if not chosen:
        worlds = dict(Counter(r.get("world") for r in rows))
        print(f"no rows match the filters; the inputs hold worlds {worlds}")
        return 0
    report(chosen)
    if args.export:
        n = export(chosen, args.export)
        print(f"\nwrote {n} labelled rows to {args.export}" if n else "\nno labelled rows to export")
    return 0


if __name__ == "__main__":
    sys.exit(main())

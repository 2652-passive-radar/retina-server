"""Rotor and propeller comb features, read from a node's detection lists.

A node integrates over a CPI of about half a second, which is many rotor
periods, so a turning blade set does not smear: it resolves into a comb of
lines at the body's delay, spaced by the blade-passing rate (blades x rotation
rate) and spread either side of the body Doppler.  The detector reports each
strong enough line as one more detection at (almost) the body's delay.  A jet
shows none, because its fan modulates in the kHz, outside any node's Doppler
window.  So the lines are what separates a jet from a turboprop where motion
cannot: both cruise at 150-250 m/s and 6-9 km.

``frame_comb`` reads one frame's lines relative to a body detection the caller
already holds; ``CombAccumulator`` summarises a track's frames.  Both are pure:
association of a body detection to a track is the tracker's job, not this
module's.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

# Blade-passing rates this module will believe.  Below the floor a spacing is
# inside one Doppler bin of the body skirt; above the ceiling fewer than three
# lines fit a node's +/-1000 Hz window, too few to call a comb.
MIN_SPACING_HZ = 5.0
MAX_SPACING_HZ = 400.0
_GRID_STEP_HZ = 0.25
# A line is on a multiple of the spacing when it is within this of it: the
# larger of a fixed floor (bin quantisation plus Doppler noise) and a fraction
# of the spacing (rpm jitter compounds with the harmonic number).
_TOL_FLOOR_HZ = 2.0
_TOL_FRACTION = 0.15
# Scores this close to the best count as ties; the largest spacing among them
# wins, because every sub-harmonic of the true spacing explains the lines as
# well as the true one does.
_SCORE_TIE = 0.05
# The detection that is the body, as opposed to a line beside it, is the one
# this close in Doppler to where the caller says the body is.
BODY_MATCH_HZ = 2.0
_MIRROR_BINS = 1.5
_DEFAULT_BIN_HZ = 2.0  # 1 / the 0.5 s CPI


@dataclass(frozen=True)
class CombObs:
    """One frame's comb around one body detection."""

    offsets_hz: tuple[float, ...]
    n_lines: int
    max_abs_offset_hz: float
    spacing_hz: float | None
    # Fraction of lines that have a mirror image about the body: a rotor seen
    # edge-on spreads symmetrically, a lopsided comb is a bistatic geometry or
    # a partial detection.  None with no lines.
    symmetry: float | None
    # Mean line SNR minus the body's, in dB; None with no lines or no body
    # detection to reference.
    mean_rel_snr_db: float | None


# A body with no line beside it: what frame_comb returns when nothing else sits
# at the body's delay, and what a caller that has already checked may use
# without calling it.
NO_COMB = CombObs((), 0, 0.0, None, None, None)


def _spacing_candidates() -> np.ndarray:
    return np.arange(MIN_SPACING_HZ, MAX_SPACING_HZ + _GRID_STEP_HZ, _GRID_STEP_HZ)


def _scores(abs_offsets: np.ndarray, spacings: np.ndarray) -> np.ndarray:
    """Per candidate spacing, the fraction of ``abs_offsets`` sitting on a nonzero multiple of it."""
    k = np.maximum(1.0, np.round(abs_offsets[None, :] / spacings[:, None]))
    resid = np.abs(abs_offsets[None, :] - k * spacings[:, None])
    tol = np.maximum(_TOL_FLOOR_HZ, _TOL_FRACTION * spacings)[:, None]
    return (resid <= tol).mean(axis=1)


def _refine(abs_offsets: np.ndarray, spacing: float) -> float:
    """Least-squares spacing over the lines assigned to its multiples.

    The grid search prefers the largest spacing among near-ties, which sits at
    the top edge of the tolerance and so reads a few percent high.  Fitting
    ``offset = k * spacing`` over the assigned lines removes that bias.
    """
    for _ in range(3):
        k = np.maximum(1.0, np.round(abs_offsets / spacing))
        on = np.abs(abs_offsets - k * spacing) <= max(_TOL_FLOOR_HZ, _TOL_FRACTION * spacing)
        if not on.any():
            break
        spacing = float((k[on] * abs_offsets[on]).sum() / (k[on] ** 2).sum())
    return spacing


def estimate_spacing(offsets_hz) -> float | None:
    """The line spacing that best explains ``offsets_hz`` as integer multiples, or None.

    None with fewer than two offsets: one line fits every spacing.
    """
    abs_offsets = np.abs(np.asarray(offsets_hz, dtype=float))
    if abs_offsets.size < 2:
        return None
    spacings = _spacing_candidates()
    scores = _scores(abs_offsets, spacings)
    near_best = np.flatnonzero(scores >= scores.max() - _SCORE_TIE)
    return _refine(abs_offsets, float(spacings[near_best[-1]]))


def _symmetry(offsets: np.ndarray, bin_hz: float) -> float:
    tol = _MIRROR_BINS * bin_hz
    paired = 0
    for i, o in enumerate(offsets):
        others = np.delete(offsets, i)
        if others.size and np.min(np.abs(others + o)) <= tol:
            paired += 1
    return paired / len(offsets)


def frame_comb(
    delays,
    dopplers,
    snrs,
    body_delay_us: float,
    body_doppler_hz: float,
    delay_tol_us: float = 1.0,
    bin_hz: float = _DEFAULT_BIN_HZ,
) -> CombObs:
    """The comb of detections sharing a body's delay in one frame.

    ``bin_hz`` is the node's Doppler resolution (1 / CPI); it sets how close a
    mirror line has to be.  The body detection itself, the one nearest
    ``body_doppler_hz`` within BODY_MATCH_HZ, is not a line and is left out.
    """
    delays = np.asarray(delays, dtype=float)
    dopplers = np.asarray(dopplers, dtype=float)
    snrs = np.asarray(snrs, dtype=float)
    at_delay = np.flatnonzero(np.abs(delays - body_delay_us) <= delay_tol_us)
    body_snr = None
    if at_delay.size:
        gap = np.abs(dopplers[at_delay] - body_doppler_hz)
        nearest = int(np.argmin(gap))
        if gap[nearest] <= BODY_MATCH_HZ:
            body_snr = float(snrs[at_delay[nearest]])
            at_delay = np.delete(at_delay, nearest)
    if not at_delay.size:
        return NO_COMB
    order = np.argsort(dopplers[at_delay] - body_doppler_hz)
    lines = at_delay[order]
    offsets = dopplers[lines] - body_doppler_hz
    rel = float(np.mean(snrs[lines] - body_snr)) if body_snr is not None else None
    return CombObs(
        offsets_hz=tuple(float(o) for o in offsets),
        n_lines=int(lines.size),
        max_abs_offset_hz=float(np.max(np.abs(offsets))),
        spacing_hz=estimate_spacing(offsets),
        symmetry=_symmetry(offsets, bin_hz),
        mean_rel_snr_db=rel,
    )


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


class CombAccumulator:
    """A track's comb observations, frame by frame.

    ``max_frames`` bounds a live track's memory: a track that lives for hours
    keeps its newest frames only, so its summary describes now rather than the
    whole flight.  The bench leaves it unbounded, as a window is short.
    """

    def __init__(self, max_frames: int | None = None):
        self._obs: deque[CombObs] = deque(maxlen=max_frames)
        self._body_snr: deque[float] = deque(maxlen=max_frames)

    def add(self, obs: CombObs, body_snr_db: float | None = None) -> None:
        self._obs.append(obs)
        if body_snr_db is not None:
            self._body_snr.append(float(body_snr_db))

    def summary(self) -> dict:
        """The track's comb in one dict; every statistic is None where nothing backs it.

        Counts and offsets are medians over the frames that showed a comb, since
        a turboprop whose lines are missed on half its frames should still read
        as the comb it has; ``frac_frames_with_comb`` carries that miss rate.
        ``spacing_spread_hz`` is the interquartile range of the per-frame
        spacings, so one harmonic-confused frame does not widen it.
        ``median_body_snr_db`` is over the frames the caller gave a body SNR for.
        """
        # list() of a deque is one C call, an atomic snapshot: the feed builder
        # reads this while a frame worker is appending.
        obs = list(self._obs)
        body_snr = list(self._body_snr)
        combs = [o for o in obs if o.n_lines > 0]
        spacings = [o.spacing_hz for o in combs if o.spacing_hz is not None]
        spread = float(np.percentile(spacings, 75) - np.percentile(spacings, 25)) if spacings else None
        return {
            "frames": len(obs),
            "frac_frames_with_comb": len(combs) / len(obs) if obs else 0.0,
            "median_n_lines": _median([o.n_lines for o in combs]),
            "max_n_lines": max((o.n_lines for o in combs), default=0),
            "median_spacing_hz": _median(spacings),
            "spacing_spread_hz": spread,
            "median_max_offset_hz": _median([o.max_abs_offset_hz for o in combs]),
            "median_symmetry": _median([o.symmetry for o in combs if o.symmetry is not None]),
            "median_rel_snr_db": _median([o.mean_rel_snr_db for o in combs if o.mean_rel_snr_db is not None]),
            "median_body_snr_db": _median(body_snr),
        }


def compact_summary(summary: dict | None) -> dict | None:
    """The four headline numbers of a summary, rounded for the feed; None with no frames."""
    if not summary or not summary.get("frames"):
        return None

    def rounded(v, nd):
        return None if v is None else round(v, nd)

    return {
        "n_lines": rounded(summary["median_n_lines"], 1),
        "spacing_hz": rounded(summary["median_spacing_hz"], 1),
        "frac_frames_with_comb": round(summary["frac_frames_with_comb"], 3),
        "frames": summary["frames"],
    }

"""Classify every target in the aircraft feed, and score the baseline.

Each feed entry gets up to two independent answers to "what is this?":

``identity``
    From the ICAO hex (services.target_identity): registration, type,
    operator, military flag and a target class.  Present only for a target an
    ADS-B transponder names.
``motion_class``
    From the target's motion alone (services.motion_classifier), the
    baseline.  Present on every entry, ``unknown`` when the motion says too
    little.

``target_class``, the feed's long-standing two-valued field, follows the
identity where there is one, because a registry entry beats any inference.
Otherwise it is left as the pipeline set it: the baseline's verdict is
published beside it, not in place of it, until the scorecard below says it has
earned that.

Wherever a target has both, the pair is a labelled example: the identity is
ground truth for the motion baseline.  ``scorecard()`` reports the confusion
matrix over each transponder's latest pair inside SCORE_WINDOW_S, one vote per
aircraft so a long-lived airliner does not outvote everything else.
"""

import logging
import threading
import time
from collections import OrderedDict

from core import state
from services import motion_classifier
from services.id_utils import is_transponder_hex, normalize_hex_key
from services.node_config import carrier_hz
from services.target_identity import TARGET_CLASSES, identify, target_class_for

log = logging.getLogger(__name__)

FT_TO_M = 0.3048
MS_PER_KNOT = 0.514444
# Sources whose trail is motion: each point a measured position of this
# target.  A single-node entry's drawn point is an arc midpoint or an
# underdetermined solve, so its trail is the delay changing, not the target
# moving.  A claimed ADS-B entry's fix is real, but it appends no trail of its
# own and the one under its hex may be a tracker track's arc midpoints, so it
# is read from its reported speed and altitude alone.
RELIABLE_POSITION_SOURCES = frozenset({"multinode_solve"})
SCORE_WINDOW_S = 6 * 3600.0
_SCORE_MAX_TARGETS = 20_000

_score_lock = threading.Lock()
_latest: "OrderedDict[str, tuple[str, str, float, float]]" = OrderedDict()


def _alt_m(alt_ft) -> float | None:
    if alt_ft == "ground":
        return 0.0
    if isinstance(alt_ft, (int, float)) and not isinstance(alt_ft, bool):
        return float(alt_ft) * FT_TO_M
    return None


def _identity_hex(entry: dict) -> str | None:
    h = normalize_hex_key(entry.get("adsb_hex") or entry.get("hex"))
    return h if is_transponder_hex(h) and not h.startswith("~") else None


def _is_simulated(entry: dict, hex_code: str) -> bool:
    """True when the target belongs to the synthetic fleet.

    A simulator draws its hexes at random, so one would look up as whatever
    real aircraft happens to own that address.  The ADS-B record's own world
    stamp decides where there is one, the reporting node's world otherwise.
    """
    rec = state.adsb_aircraft.get(hex_code)
    if rec is not None and rec.get("world"):
        return rec["world"] == "sim"
    node_id = entry.get("node_id")
    if node_id:
        return state.node_world(node_id) == "sim"
    nodes = entry.get("contributing_node_ids") or []
    return bool(nodes) and all(state.node_world(n) == "sim" for n in nodes)


def _emitter_category(hex_code: str) -> str | None:
    for store in (state.adsb_aircraft, state.external_adsb_cache):
        rec = store.get(hex_code)
        if rec and rec.get("category"):
            return rec["category"]
    return None


def _fc_hz(node_id: str | None) -> float | None:
    pipeline = state.node_pipelines.get(node_id) if node_id else None
    fc = carrier_hz(getattr(pipeline, "config", None) or {})
    return float(fc) if isinstance(fc, (int, float)) and fc > 0 else None


def _trail(entry: dict) -> list[tuple[float, float, float, float | None]]:
    hist = state.track_histories.get(entry.get("hex"))
    if not hist:
        return []
    out = []
    for p in list(hist):
        try:
            lat, lon, alt_ft, ts = p[0], p[1], p[2], p[3]
        except (IndexError, TypeError):
            continue
        out.append((float(ts), float(lat), float(lon), _alt_m(alt_ft)))
    return out


def motion_for(entry: dict) -> motion_classifier.MotionClass:
    """The baseline's verdict on one feed entry."""
    alt = _alt_m(entry.get("alt_baro"))
    source = entry.get("position_source")
    gs = entry.get("gs")
    speed = gs * MS_PER_KNOT if isinstance(gs, (int, float)) and not isinstance(gs, bool) else None
    if source in RELIABLE_POSITION_SOURCES:
        return motion_classifier.classify(_trail(entry), alt_m=alt, speed_ms=speed)
    if source == "adsb_single_node":
        return motion_classifier.classify(None, alt_m=alt, speed_ms=speed)
    floor = motion_classifier.doppler_speed_floor_ms(entry.get("doppler_hz"), _fc_hz(entry.get("node_id")))
    return motion_classifier.classify(None, speed_floor_ms=floor, alt_m=alt)


def _record(hex_code: str, truth: str, predicted: str, confidence: float, now: float) -> None:
    with _score_lock:
        _latest.pop(hex_code, None)
        _latest[hex_code] = (truth, predicted, confidence, now)
        cutoff = now - SCORE_WINDOW_S
        while _latest:
            oldest = next(iter(_latest.values()))
            if oldest[3] >= cutoff and len(_latest) <= _SCORE_MAX_TARGETS:
                break
            _latest.popitem(last=False)


def classify_entry(entry: dict, now: float | None = None) -> dict:
    """Annotate one feed entry in place with ``identity`` and ``motion_class``."""
    now = time.time() if now is None else now
    hex_code = _identity_hex(entry)
    identity = None
    if hex_code is not None:
        cat = _emitter_category(hex_code)
        identity = identify(None if _is_simulated(entry, hex_code) else hex_code, cat)
    motion = motion_for(entry)
    entry["motion_class"] = motion.as_dict()
    if identity is not None:
        entry["identity"] = identity
        legacy = target_class_for(identity.get("class"))
        if legacy is not None:
            entry["target_class"] = legacy
        if identity.get("class") in TARGET_CLASSES:
            _record(hex_code, identity["class"], motion.cls, motion.confidence, now)
    return entry


def classify_feed(aircraft: list[dict], now: float | None = None) -> list[dict]:
    """Annotate every entry; one that fails is published unclassified.

    Classification is enrichment, so it must never cost the broadcast the way
    an exception escaping the feed build would (see aircraft_feed's
    _note_multinode_entry_failure for the same rule).
    """
    now = time.time() if now is None else now
    for entry in aircraft:
        try:
            classify_entry(entry, now)
        except Exception:
            state.bump_counter("target_classification_errors")
            log.debug("classification failed for %s", entry.get("hex"), exc_info=True)
    return aircraft


def scorecard(now: float | None = None) -> dict:
    """The baseline against the hex label, one vote per aircraft.

    ``accuracy`` is over the targets the baseline committed on; ``coverage``
    is how many it committed on at all.  Both None with nothing to score.
    """
    now = time.time() if now is None else now
    cutoff = now - SCORE_WINDOW_S
    with _score_lock:
        rows = [v for v in _latest.values() if v[3] >= cutoff]
    labels = list(TARGET_CLASSES) + [motion_classifier.UNKNOWN]
    confusion = {t: {p: 0 for p in labels} for t in TARGET_CLASSES}
    for truth, predicted, _, _ in rows:
        confusion[truth][predicted if predicted in confusion[truth] else motion_classifier.UNKNOWN] += 1
    decided = [(t, p) for t, p, _, _ in rows if p != motion_classifier.UNKNOWN]
    correct = sum(1 for t, p in decided if t == p)
    per_class = {}
    for cls in TARGET_CLASSES:
        n = sum(confusion[cls].values())
        n_decided = n - confusion[cls][motion_classifier.UNKNOWN]
        predicted_as = sum(confusion[t][cls] for t in TARGET_CLASSES)
        per_class[cls] = {
            "targets": n,
            "recall": round(confusion[cls][cls] / n_decided, 3) if n_decided else None,
            "precision": round(confusion[cls][cls] / predicted_as, 3) if predicted_as else None,
        }
    return {
        "window_s": SCORE_WINDOW_S,
        "targets": len(rows),
        "decided": len(decided),
        "accuracy": round(correct / len(decided), 3) if decided else None,
        "coverage": round(len(decided) / len(rows), 3) if rows else None,
        "per_class": per_class,
        "confusion": confusion,
    }


def _reset_for_tests() -> None:
    with _score_lock:
        _latest.clear()

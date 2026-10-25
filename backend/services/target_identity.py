"""Label a target from its ICAO hex: what the airframe is, and its class.

Two sources, best first:

1. The aircraft registry (services.aircraft_db): registration, ICAO type
   designator, operator and the ICAO aircraft class of that type.  This is the
   authoritative answer whenever the hex is in it.
2. The ADS-B emitter category the transponder broadcasts (``A3``, ``A7``,
   ``B6``...), where a record carries one.  Coarser, but self-reported by the
   aircraft and present for addresses the registry has never seen.

Both are reduced to the same small vocabulary of TARGET_CLASSES that the motion
baseline (services.motion_classifier) predicts, so the two can be scored
against each other: the label is the ground truth the baseline is measured by.
"""

from services import aircraft_db

# Ordered by how a performance envelope separates them, which is all a
# motion-only classifier can see: jets and heavy turboprops fly fast and high,
# light fixed wing slower and lower, rotorcraft and drones can hover.
FAST_FIXED_WING = "fast_fixed_wing"
LIGHT_FIXED_WING = "light_fixed_wing"
ROTORCRAFT = "rotorcraft"
GLIDER = "glider"
BALLOON = "balloon"
DRONE = "drone"
SURFACE = "surface"
TARGET_CLASSES = (FAST_FIXED_WING, LIGHT_FIXED_WING, ROTORCRAFT, GLIDER, BALLOON, DRONE)

# ADS-B emitter category (DO-260B, as tar1090 spells it) → class.  A0/B0/C0
# mean "no information" and B3 (parachutist), B5 (reserved) and B7 (space
# vehicle) have no class here, so they are left out and label nothing.
_EMITTER_CATEGORY = {
    "A1": LIGHT_FIXED_WING,
    "A2": FAST_FIXED_WING,
    "A3": FAST_FIXED_WING,
    "A4": FAST_FIXED_WING,
    "A5": FAST_FIXED_WING,
    "A6": FAST_FIXED_WING,
    "A7": ROTORCRAFT,
    "B1": GLIDER,
    "B2": BALLOON,
    "B4": LIGHT_FIXED_WING,
    "B6": DRONE,
    "C1": SURFACE,
    "C2": SURFACE,
    "C3": SURFACE,
}

# Type designators whose ICAO class (``L0-`` and the like) says too little.
_TYPE_CODE = {
    "GLID": GLIDER,
    "ULAC": LIGHT_FIXED_WING,
    "BALL": BALLOON,
    "SHIP": BALLOON,
    "DRON": DRONE,
    "UAV": DRONE,
    "GYRO": ROTORCRAFT,
    "TWR": SURFACE,
    "GND": SURFACE,
}


def class_from_icao(icao_class: str | None, wtc: str | None = None) -> str | None:
    """Reduce an ICAO aircraft class (``L2J``, ``H1T``...) to a target class.

    The first letter is the airframe: Landplane, Seaplane, Amphibian,
    Helicopter, Gyrocopter, Tiltrotor, Drone, Balloon/airship, ground Vehicle.
    The last is the engine: Jet, Turboprop, Piston, Electric.  A fixed-wing jet
    is fast whatever its size; a turboprop is fast once it is medium wake or
    larger (an ATR, not a Caravan); everything else fixed-wing is light.
    """
    if not icao_class:
        return None
    kind = icao_class[0].upper()
    if kind in ("H", "G", "T", "R"):
        return ROTORCRAFT
    if kind == "D":
        return DRONE
    if kind == "B":
        return BALLOON
    if kind == "V":
        return SURFACE
    if kind not in ("L", "S", "A"):
        return None
    engine = icao_class[-1].upper() if len(icao_class) >= 3 else ""
    engines = icao_class[1] if len(icao_class) >= 2 else ""
    if engine == "J":
        return FAST_FIXED_WING
    heavy = (wtc or "").upper() in ("M", "H", "J", "L/M")
    if engine == "T" and (heavy or (engines.isdigit() and int(engines) >= 2 and not wtc)):
        return FAST_FIXED_WING
    if engine in ("T", "P", "E"):
        return LIGHT_FIXED_WING
    return None


def _flag(flags: str | None, i: int) -> bool:
    # tar1090-db writes dbFlags as a digit string: military, interesting, PIA, LADD.
    return bool(flags) and len(flags) > i and flags[i] == "1"


def _is_military(flags: str | None) -> bool:
    return _flag(flags, 0)


def _owner_asked_for_privacy(flags: str | None) -> bool:
    """FAA PIA or LADD: the owner asked for this aircraft not to be identified."""
    return _flag(flags, 2) or _flag(flags, 3)


def identify(hex_code, emitter_category: str | None = None) -> dict | None:
    """The identity label for one target, or None when nothing names it.

    ``class`` is None when a record names the airframe but nothing maps it to
    a class (a type with no ICAO class); the descriptive fields still stand.
    """
    cat = (emitter_category or "").strip().upper()
    rec = aircraft_db.lookup(hex_code)
    if rec is not None:
        type_code = rec.get("type_code")
        cls = _TYPE_CODE.get(type_code or "") or class_from_icao(rec.get("icao_class"), rec.get("wtc"))
        label = {
            "source": "aircraft_db",
            "class": cls,
            "military": _is_military(rec.get("flags")),
        }
        for out, key in (
            ("registration", "registration"),
            ("type_code", "type_code"),
            ("type_name", "type_name"),
            ("description", "description"),
            ("icao_class", "icao_class"),
            ("wtc", "wtc"),
            ("operator", "operator"),
        ):
            if rec.get(key):
                label[out] = rec[key]
        # The feed is public, so a privacy-listed aircraft keeps its type and
        # class but not the fields that name who flies it.
        if _owner_asked_for_privacy(rec.get("flags")):
            label.pop("registration", None)
            label.pop("operator", None)
            label["privacy"] = True
        if cls is None and cat in _EMITTER_CATEGORY:
            label["class"] = _EMITTER_CATEGORY[cat]
        if rec.get("registration") or type_code or label["class"] or label["military"]:
            return label
    if cat in _EMITTER_CATEGORY:
        return {"source": "adsb_category", "class": _EMITTER_CATEGORY[cat], "category": cat, "military": False}
    return None


def target_class_for(cls: str | None) -> str | None:
    """The feed's legacy two-valued ``target_class`` for a class, or None."""
    if cls is None or cls == SURFACE:
        return None
    return "drone" if cls == DRONE else "aircraft"

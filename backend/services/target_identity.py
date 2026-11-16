"""Label a target from its ICAO hex: what the airframe is, and its class.

Two sources, best first:

1. The aircraft registry (services.aircraft_db): registration, ICAO type
   designator, operator and the ICAO aircraft class of that type.  This is the
   authoritative answer whenever the hex is in it.
2. The ADS-B emitter category the transponder broadcasts (``A7``, ``B6``...),
   where a record carries one.  Coarser, but self-reported by the aircraft and
   present for addresses the registry has never seen.

Both are reduced to the same small vocabulary that the motion baseline
(services.motion_classifier) predicts, so the two can be scored against each
other: the label is the ground truth the baseline is measured by.

The classes are bins of what produces the Doppler signature, not of airframe
shape: a jet's compressor and turbine blades, a prop's blades, a rotor's
blades, or nothing but a small body.  The end goal is to classify on Doppler
and micro-Doppler, so that is the axis the labels have to follow.
"""

from services import aircraft_db

JET = "jet"
TURBOPROP = "turboprop"
PISTON = "piston"
HELICOPTER = "helicopter"
DRONE = "drone"
# Identity only: gliders, balloons, airships, gyrocopters, tiltrotors, ground
# vehicles and towers.  Motion has no envelope for it, so the baseline never
# predicts it and the scorecard does not record targets labelled with it.
OTHER = "other"
# What a motion-only classifier is asked to predict, and the scorecard scores.
TARGET_CLASSES = (JET, TURBOPROP, PISTON, HELICOPTER, DRONE)

# ADS-B emitter category (DO-260B, as tar1090 spells it) → class.  A1-A6 are
# wake/weight classes (light, small, large, high-vortex, heavy, high
# performance) and say nothing about the engine: a "small" A2 can be a
# turboprop, a jet or a piston twin, so they cannot be told apart by jet vs
# turboprop vs piston and are deliberately left out rather than guessed.
# A0/B0/C0 mean "no information" and B5 (reserved) has no class, so they label
# nothing.
_EMITTER_CATEGORY = {
    "A7": HELICOPTER,
    "B1": OTHER,
    "B2": OTHER,
    "B3": OTHER,
    "B4": PISTON,
    "B6": DRONE,
    "B7": OTHER,
    "C1": OTHER,
    "C2": OTHER,
    "C3": OTHER,
}

# Type designators whose ICAO class (``L0-`` and the like) says too little.
_TYPE_CODE = {
    "DRON": DRONE,
    "UAV": DRONE,
    "GLID": OTHER,
    "BALL": OTHER,
    "SHIP": OTHER,
    "TWR": OTHER,
    "GND": OTHER,
    "GYRO": OTHER,
    "ULAC": PISTON,
}

# Propeller or main-rotor blade count by ICAO type designator.  Values are for
# the common variant of each type, from general knowledge, and should be
# verified before anything leans on them.  The blade count sets the spacing of
# the micro-Doppler lines (blade-passing frequency = blades x rotation rate),
# so it is the one airframe fact a Doppler classifier wants beyond the class.
BLADES = {
    "DH8D": 6,
    "DH8A": 4,
    "DH8B": 4,
    "DH8C": 4,
    "AT72": 6,
    "AT76": 6,
    "AT45": 6,
    "AT75": 6,
    "SF34": 4,
    "DHC6": 3,
    "C172": 2,
    "C152": 2,
    "PA28": 2,
    "R44": 2,
    "R22": 2,
    "AS50": 3,
    "EC35": 4,
    "S76": 4,
    "B06": 2,
    "EC30": 3,
    "A139": 5,
    "B412": 4,
}


def class_from_icao(icao_class: str | None) -> str | None:
    """Reduce an ICAO aircraft class (``L2J``, ``H1T``...) to a target class.

    The first letter is the airframe: Landplane, Seaplane, Amphibian,
    Helicopter, Gyrocopter, Tiltrotor, Drone, Balloon/airship, ground Vehicle.
    The last is the engine: Jet, Turboprop, Piston, Electric.  A fixed-wing
    aircraft is binned by its engine, since that is what its Doppler shows.
    """
    if not icao_class:
        return None
    kind = icao_class[0].upper()
    if kind == "H":
        return HELICOPTER
    if kind == "D":
        return DRONE
    if kind not in ("L", "S", "A"):
        # G gyrocopter, T tiltrotor, R rocket, B balloon/airship, V vehicle.
        return OTHER
    engine = icao_class[-1].upper() if len(icao_class) >= 3 else ""
    if engine == "J":
        return JET
    if engine == "T":
        return TURBOPROP
    # Electric is prop-driven (Harbour Air's electric Beaver), so it shows the
    # same blade lines as a piston engine.
    if engine in ("P", "E"):
        return PISTON
    return OTHER


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
        cls = _TYPE_CODE.get(type_code or "") or class_from_icao(rec.get("icao_class"))
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
        if type_code in BLADES:
            label["blades"] = BLADES[type_code]
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
    if cls is None or cls == OTHER:
        return None
    return "drone" if cls == DRONE else "aircraft"

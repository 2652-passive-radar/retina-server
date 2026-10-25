"""Keep the aircraft registry (services.aircraft_db) built and current.

Runs only where AIRCRAFT_DB_ENABLED is exactly `1`, like every other task that
reaches out to a third party.  It builds the registry at boot if there is none,
then rebuilds it once it is older than MAX_AGE_S: tar1090-db republishes daily,
and a registration or type change a week late labels nothing wrongly enough to
be worth more traffic than that.  Anywhere it is off, a registry file put in
place by ``scripts/fetch_aircraft_db.py`` is still read.
"""

import asyncio
import logging
import os
import time

from core import state
from services import aircraft_db

log = logging.getLogger(__name__)

ENABLED_ENV = "AIRCRAFT_DB_ENABLED"
MAX_AGE_S = 7 * 86400.0
CHECK_INTERVAL_S = 3600.0
RETRY_S = 900.0

_TASK = "aircraft_db_refresh"


def enabled() -> bool:
    return os.getenv(ENABLED_ENV) == "1"


def due() -> bool:
    age = aircraft_db.age_s()
    return age is None or age > MAX_AGE_S


async def aircraft_db_refresh_task() -> None:
    """The lifespan's task: rebuild when due, or return at once where it is off."""
    if not enabled():
        log.info("aircraft registry is not downloaded (%s is not 1)", ENABLED_ENV)
        return
    while True:
        wait = CHECK_INTERVAL_S
        try:
            if due():
                await asyncio.to_thread(aircraft_db.refresh)
            state.task_last_success[_TASK] = time.time()
        except Exception:
            state.bump_task_error(_TASK)
            log.exception("aircraft registry refresh failed; retrying in %.0fs", RETRY_S)
            wait = RETRY_S
        await asyncio.sleep(wait)

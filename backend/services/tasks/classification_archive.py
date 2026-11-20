"""Background task: write the classification archive's buffered rows to Parquet, hourly.

The rows are built by services.classification_archive as the aircraft feed is
built; this only drains them to disk.  It runs on the hour's cadence rather than
the track archive's minute because the archive's unit is an hourly file, and a
flush that lands mid-hour merges into the hour's file anyway.  A restart loses
at most the rows buffered since the last flush; the shutdown path flushes what
is there (main.py).
"""

import asyncio
import logging
import time

from config.constants import CLASSIFICATION_ARCHIVE_FLUSH_INTERVAL_S
from core import state
from services import classification_archive

logger = logging.getLogger(__name__)


async def classification_archive_flush_task():
    """Periodic flush loop.  Runs forever as a background asyncio task."""
    while True:
        await asyncio.sleep(CLASSIFICATION_ARCHIVE_FLUSH_INTERVAL_S)
        try:
            # The write is disk and Parquet encoding: off the event loop.
            files = await asyncio.to_thread(classification_archive.flush)
            if files:
                logger.info("classification archive flushed: %s", ", ".join(files))
            state.task_last_success["classification_archive_flush"] = time.time()
        except Exception:
            state.bump_task_error("classification_archive_flush")
            logger.exception("classification_archive_flush_task iteration failed")

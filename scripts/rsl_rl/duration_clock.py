"""Day-aware ``Time elapsed:`` / ``ETA:`` in the rsl-rl console printout.

rsl-rl renders both as ``time.strftime("%H:%M:%S", time.gmtime(seconds))``
(``rsl_rl/utils/logger.py``, in the footer block). ``gmtime`` converts a duration as if it were a
wall-clock instant, so ``%H`` is the hour-of-day: at 24 h it silently rolls back to ``00:``. A
26-hour run reports ``02:00:00`` elapsed, and a full-profile walk run is long enough to hit that.

Days are not recoverable after the fact -- ``%j`` would give a day-of-year starting at 1, so the
existing format simply drops the information. The fix has to happen where the duration is still a
number of seconds.

``time`` is referenced **only** on those two lines in that module (verified against rsl-rl 5.0.1),
so swapping the module's ``time`` binding for this shim cannot affect anything else. Everything
except the ``gmtime``/``strftime`` pair falls through to the real module, so a future rsl-rl that
starts calling ``time.time()`` there keeps working.

Under 24 h the output is byte-identical to before (``HH:MM:SS``); past it, a day count is
prefixed (``1d 02:00:00``).
"""

from __future__ import annotations

import time as _time


class _Duration:
    """Marker returned by the shim's ``gmtime`` carrying the raw seconds.

    Exists so the paired ``strftime`` can tell "this is a duration rsl-rl is about to render"
    from any other struct_time it might be handed.
    """

    __slots__ = ("seconds",)

    def __init__(self, seconds: float) -> None:
        self.seconds = max(0.0, float(seconds))

    def render(self) -> str:
        total = int(self.seconds)
        days, rem = divmod(total, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        stamp = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{days}d {stamp}" if days else stamp


class _DayAwareTime:
    """Stand-in for the ``time`` module, used only by ``rsl_rl.utils.logger``."""

    @staticmethod
    def gmtime(seconds=None):
        # rsl-rl only ever passes a duration here. A bare gmtime() is a real wall-clock call.
        if seconds is None:
            return _time.gmtime()
        return _Duration(seconds)

    @staticmethod
    def strftime(fmt, value=None):
        if isinstance(value, _Duration):
            return value.render()
        return _time.strftime(fmt) if value is None else _time.strftime(fmt, value)

    def __getattr__(self, name):
        # Anything the logger might use in future (time.time, monotonic, ...).
        return getattr(_time, name)


def install() -> None:
    """Swap the ``time`` binding inside rsl-rl's logger. Idempotent."""
    from rsl_rl.utils import logger as _logger

    if not isinstance(getattr(_logger, "time", None), _DayAwareTime):
        _logger.time = _DayAwareTime()

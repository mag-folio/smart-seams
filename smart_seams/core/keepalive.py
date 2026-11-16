"""Keep Blender's window from turning "Not Responding" during a long solve.

Windows greys a window out and marks it *Not Responding* when its thread has
not looked at its message queue for about five seconds. A dense mesh keeps
the solver busy for much longer than that. ``pulse()`` peeks at the queue
(PM_NOREMOVE: nothing is taken out or dispatched to Blender) at most once a
second, which is enough for Windows to see the window as alive. The picture
is not redrawn; the call is cheap and does nothing on other systems or in
background mode.
"""

import sys
import time

_last = 0.0
_peek = None

if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes

        _user32 = ctypes.windll.user32
        _msg = wintypes.MSG()
        _PM_NOREMOVE_NOYIELD = 0x0000 | 0x0002

        def _peek():
            _user32.PeekMessageW(ctypes.byref(_msg), None, 0, 0, _PM_NOREMOVE_NOYIELD)
    except Exception:  # no ctypes / user32: nothing to keep alive
        _peek = None


def pulse(interval=1.0):
    """Tell Windows the window is alive (throttled to ``interval`` seconds)."""
    global _last
    if _peek is None:
        return
    now = time.monotonic()
    if now - _last < interval:
        return
    _last = now
    try:
        _peek()
    except Exception:
        pass

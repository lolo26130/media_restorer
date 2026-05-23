"""Power management: switches to performance profile during restoration.

Uses ``power-profiles-daemon`` (via ``powerprofilesctl``) to switch the
system to the ``performance`` profile without requiring root privileges.
Restores the original profile on exit, even on exception.

Also inhibits system sleep via ``systemd-inhibit`` for the duration.

Usage::

    from media_restorer.power import performance_mode

    with performance_mode():
        result = restore_image(...)
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager


def _get_profile() -> str | None:
    try:
        result = subprocess.run(
            ["powerprofilesctl", "get"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _set_profile(profile: str) -> bool:
    try:
        result = subprocess.run(
            ["powerprofilesctl", "set", profile],
            capture_output=True, timeout=3,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _start_inhibit() -> subprocess.Popen | None:
    """Hold a systemd sleep inhibitor lock for the duration of the process."""
    try:
        return subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=sleep:idle",
                "--who=media-restorer",
                "--why=Restauration en cours",
                "--mode=block",
                "sleep", "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None


def _stop_inhibit(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def performance_mode(verbose: bool = True):
    """Context manager that boosts CPU/system to *performance* profile.

    Parameters
    ----------
    verbose:
        Print status messages to stdout when ``True``.

    Examples
    --------
    >>> with performance_mode():
    ...     restore_image("old.jpg", "restored.jpg")
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"[power] {msg}", flush=True)

    original = _get_profile()

    profile_ok = False
    if original is None:
        log("power-profiles-daemon absent — profil inchangé")
    elif original == "performance":
        log("Profil déjà en 'performance'")
        profile_ok = True  # already set, still restore to same value on exit
    else:
        profile_ok = _set_profile("performance")
        if profile_ok:
            log(f"Profil CPU : '{original}' → 'performance'")
        else:
            log("Impossible de changer le profil (continuera en mode normal)")

    inhibit_proc = _start_inhibit()
    if inhibit_proc is not None:
        log("Veille système désactivée")
    else:
        log("systemd-inhibit absent — veille non bloquée")

    try:
        yield
    finally:
        _stop_inhibit(inhibit_proc)
        if inhibit_proc is not None:
            log("Veille système réactivée")

        if profile_ok and original is not None and original != "performance":
            if _set_profile(original):
                log(f"Profil CPU : 'performance' → '{original}' (restauré)")
            else:
                log(f"Avertissement : impossible de restaurer le profil '{original}'")

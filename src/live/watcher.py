"""
Streaming log watchers for live Hearthstone sessions and mock replay playback.
"""

from __future__ import annotations

import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Generator, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_HS_LOG_DIRS = [
    Path(r"D:\Hearthstone\Logs"),
    Path(os.path.expandvars(r"%LOCALAPPDATA%\Blizzard\Hearthstone\Logs")),
]


class HearthstoneLogWatcher:
    """
    Watches live Hearthstone logs in D:\\Hearthstone\\Logs or local appdata.
    Auto-discovers the newest session directory (Hearthstone_YYYY_MM_DD_HH_MM_SS)
    and streams newly appended lines from Power.log, handling file rotation
    and game restarts.
    """

    def __init__(
        self,
        log_dir: Optional[Path | str] = None,
        catch_up: bool = True,
    ):
        self.base_dir = self._resolve_base_dir(log_dir)
        self.catch_up = catch_up
        self.current_session_dir: Optional[Path] = None
        self.current_file_path: Optional[Path] = None
        self._last_offset: int = 0
        self._last_inode_or_mtime: float = 0.0

    @staticmethod
    def _resolve_base_dir(custom_path: Optional[Path | str]) -> Path:
        if custom_path:
            p = Path(custom_path)
            if p.exists():
                return p
        for default_dir in DEFAULT_HS_LOG_DIRS:
            if default_dir.exists():
                return default_dir
        return DEFAULT_HS_LOG_DIRS[0]

    def find_latest_session_dir(self) -> Optional[Path]:
        """Finds the most recently modified Hearthstone session folder."""
        if not self.base_dir.exists():
            return None
        dirs = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("Hearthstone_")
        ]
        if not dirs:
            # Fallback to any directory or base dir itself if logs are flat
            dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        if not dirs:
            return None
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return dirs[0]

    def find_active_power_log(self) -> Optional[Path]:
        """Locates the active Power.log (or Power_old.log) in the newest session."""
        session_dir = self.find_latest_session_dir()
        if not session_dir:
            # Check if Power.log is directly in base_dir
            flat_power = self.base_dir / "Power.log"
            if flat_power.exists():
                return flat_power
            return None

        power_log = session_dir / "Power.log"
        if power_log.exists():
            return power_log

        power_old = session_dir / "Power_old.log"
        if power_old.exists():
            return power_old

        return None

    def stream_lines(
        self,
        poll_interval: float = 0.1,
        max_idle_polls: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Continuously yields new lines from Hearthstone's Power.log.
        Seamlessly handles game client restarts and session directory changes.
        """
        active_file = self.find_active_power_log()
        idle_polls = 0

        while active_file is None:
            time.sleep(poll_interval)
            active_file = self.find_active_power_log()
            idle_polls += 1
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                return

        self.current_file_path = active_file
        self.current_session_dir = active_file.parent
        logger.info("Attached to Hearthstone log: %s", active_file)

        # Open file
        f = open(active_file, "r", encoding="utf-8", errors="replace")
        try:
            if not self.catch_up:
                # Seek to end of file to only read future events
                f.seek(0, os.SEEK_END)
            self._last_offset = f.tell()

            while True:
                line = f.readline()
                if line:
                    self._last_offset = f.tell()
                    idle_polls = 0
                    yield line
                else:
                    # Check for session directory rotation (new game launched)
                    latest_session = self.find_latest_session_dir()
                    if latest_session and latest_session != self.current_session_dir:
                        new_log = latest_session / "Power.log"
                        if new_log.exists() and new_log.stat().st_size > 0:
                            logger.info("New Hearthstone session detected: %s", latest_session.name)
                            f.close()
                            active_file = new_log
                            self.current_session_dir = latest_session
                            self.current_file_path = new_log
                            f = open(active_file, "r", encoding="utf-8", errors="replace")
                            self._last_offset = 0
                            continue

                    # Check if file was truncated/recreated in same dir
                    if active_file.exists():
                        current_size = active_file.stat().st_size
                        if current_size < self._last_offset:
                            logger.info("Log file truncated, resetting offset")
                            f.seek(0, os.SEEK_SET)
                            self._last_offset = 0

                    time.sleep(poll_interval)
                    idle_polls += 1
                    if max_idle_polls is not None and idle_polls >= max_idle_polls:
                        break
        finally:
            f.close()


class MockLogWatcher:
    """
    Mock watcher for offline testing, benchmarks, and CI replay playback.
    Supports .hdtreplay zip archives, raw Power.log files, or in-memory line streams.
    """

    def __init__(
        self,
        source: Path | str | Iterable[str],
        speed: float = 0.0,
    ):
        self.source = source
        self.speed = speed

    def stream_lines(self) -> Generator[str, None, None]:
        if isinstance(self.source, (list, tuple)):
            for line in self.source:
                if self.speed > 0:
                    time.sleep(0.001 / self.speed)
                yield line
            return

        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        if path.suffix.lower() == ".hdtreplay":
            # Extract output_log.txt from zip
            with zipfile.ZipFile(path, "r") as zf:
                target_name = None
                for name in zf.namelist():
                    if name.endswith("output_log.txt") or name.endswith("Power.log"):
                        target_name = name
                        break
                if not target_name and zf.namelist():
                    target_name = zf.namelist()[0]

                if not target_name:
                    return

                with zf.open(target_name) as raw_file:
                    for raw_line in raw_file:
                        text = raw_line.decode("utf-8", errors="replace")
                        if "\\n" in text and text.count("\n") <= 1:
                            text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
                            for line in text.splitlines(keepends=True):
                                if self.speed > 0:
                                    time.sleep(0.001 / self.speed)
                                yield line
                        else:
                            if self.speed > 0:
                                time.sleep(0.001 / self.speed)
                            yield text
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if self.speed > 0:
                        time.sleep(0.001 / self.speed)
                    yield line

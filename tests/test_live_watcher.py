import os
import tempfile
import time
from pathlib import Path
import pytest

from src.live.watcher import HearthstoneLogWatcher, MockLogWatcher


def test_hearthstone_log_watcher_discovery():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        sess1 = base / "Hearthstone_2026_09_25_10_00_00"
        sess1.mkdir()
        (sess1 / "Power.log").write_text("line1\n", encoding="utf-8")

        sess2 = base / "Hearthstone_2026_09_28_12_00_00"
        sess2.mkdir()
        p2 = sess2 / "Power.log"
        p2.write_text("game_start\n", encoding="utf-8")

        watcher = HearthstoneLogWatcher(log_dir=base, catch_up=True)
        latest = watcher.find_latest_session_dir()
        assert latest is not None
        assert latest.name == "Hearthstone_2026_09_28_12_00_00"

        active = watcher.find_active_power_log()
        assert active is not None
        assert active == p2


def test_hearthstone_log_watcher_streaming():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        sess = base / "Hearthstone_2026_09_28_15_00_00"
        sess.mkdir()
        p = sess / "Power.log"
        p.write_text("Line 1\nLine 2\n", encoding="utf-8")

        watcher = HearthstoneLogWatcher(log_dir=base, catch_up=True)
        # Read with max_idle_polls to prevent infinite loop
        lines = list(watcher.stream_lines(poll_interval=0.01, max_idle_polls=3))
        assert len(lines) == 2
        assert lines[0].strip() == "Line 1"
        assert lines[1].strip() == "Line 2"


def test_mock_log_watcher_iterable():
    lines_in = ["Line 1\n", "Line 2\n", "Line 3\n"]
    watcher = MockLogWatcher(source=lines_in, speed=0.0)
    lines_out = list(watcher.stream_lines())
    assert lines_out == lines_in


def test_mock_log_watcher_file():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write("Line A\nLine B\n")
        f_path = f.name

    try:
        watcher = MockLogWatcher(source=f_path, speed=0.0)
        lines = list(watcher.stream_lines())
        assert len(lines) == 2
        assert lines[0] == "Line A\n"
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)

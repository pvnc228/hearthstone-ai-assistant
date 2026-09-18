import pytest

from src.llm.resource_guard import (
    ResourceBudget,
    ResourceGuard,
    ResourceLimitExceeded,
    ResourceSnapshot,
    conservative_environment,
    parse_nvidia_smi_row,
    unsafe_reasons,
)


def _snapshot(*, gpu_used=1000, gpu_free=7000, ram_free=16000, temperature=45):
    return ResourceSnapshot(
        gpu_total_mib=8188,
        gpu_used_mib=gpu_used,
        gpu_free_mib=gpu_free,
        gpu_temperature_c=temperature,
        gpu_utilization_percent=10,
        ram_total_mib=32600,
        ram_available_mib=ram_free,
    )


def test_default_budget_rejects_the_previous_batch_size():
    budget = ResourceBudget()

    with pytest.raises(ResourceLimitExceeded, match="batch_size=32"):
        budget.validate_batch_size(32)

    budget.validate_batch_size(2)


def test_environment_caps_cpu_and_disables_tokenizer_parallelism():
    environment = conservative_environment(ResourceBudget())

    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_watchdog_waits_for_resources_then_continues():
    snapshots = iter([_snapshot(gpu_used=6000, gpu_free=2000), _snapshot()])
    sleeps = []
    times = iter([0.0, 1.0])
    guard = ResourceGuard(
        ResourceBudget(pause_poll_seconds=0.01),
        snapshot_provider=lambda: next(snapshots),
        sleeper=sleeps.append,
        clock=lambda: next(times),
    )

    assert guard.wait_until_safe() == _snapshot()
    assert sleeps == [0.01]


def test_watchdog_aborts_instead_of_waiting_forever():
    times = iter([0.0, 61.0])
    guard = ResourceGuard(
        ResourceBudget(max_pause_seconds=60.0),
        snapshot_provider=lambda: _snapshot(ram_free=2000),
        sleeper=lambda _: None,
        clock=lambda: next(times),
    )

    with pytest.raises(ResourceLimitExceeded, match="RAM free"):
        guard.wait_until_safe()


def test_unsafe_reasons_cover_gpu_ram_and_temperature():
    reasons = unsafe_reasons(
        _snapshot(gpu_used=5000, gpu_free=3000, ram_free=4000, temperature=80),
        ResourceBudget(),
    )

    assert len(reasons) == 4


def test_nvidia_smi_parser_is_strict():
    assert parse_nvidia_smi_row("8188, 620, 7568, 41, 5") == (8188, 620, 7568, 41, 5)
    with pytest.raises(ValueError):
        parse_nvidia_smi_row("8188, 620")


def test_preflight_uses_live_telemetry_and_tracks_extremes():
    guard = ResourceGuard(
        ResourceBudget(),
        snapshot_provider=lambda: _snapshot(gpu_used=1200, gpu_free=6988, ram_free=15000),
    )

    assert guard.preflight().gpu_used_mib == 1200
    assert guard.telemetry_summary() == {
        "samples": 1,
        "max_gpu_used_mib": 1200,
        "min_gpu_free_mib": 6988,
        "min_ram_available_mib": 15000,
        "max_gpu_temperature_c": 45,
    }

def pytest_configure(config):  # pragma: no cover - pytest hook
    config._lux_samples = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # pragma: no cover - pytest hook
    samples = getattr(config, "_lux_samples", [])
    if not samples:
        return
    terminalreporter.section("Luxembourgish phonemizer samples")
    for phrase, phonemes in samples:
        terminalreporter.line(f"{phrase} -> {phonemes}")

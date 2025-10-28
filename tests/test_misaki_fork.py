"""
Test that Kokoro is wired against the Luxembourgish-enabled Misaki fork.

The upstream Misaki build on PyPI lacks `misaki.lb`, so if Kokoro can
instantiate the Luxembourgish pipeline we know the fork is active.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kokoro import KPipeline


def test_kokoro_uses_luxembourgish_misaki():
    """Ensure the active Misaki provides Luxembourgish support used by Kokoro."""
    lb = importlib.import_module("misaki.lb")
    g2p = lb.LBG2P()
    assert g2p.lexicon, "Luxembourgish dictionary failed to load from misaki.lb"

    pipeline = KPipeline(lang_code="l", model=False)
    assert isinstance(pipeline.g2p, lb.LBG2P)

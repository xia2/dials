from __future__ import annotations


def test_gltbx_is_available():
    """
    This is not a real test for dials.rlv, which is slightly difficult to write.
    However, one common error mode is that the gltbx libraries are not available
    because they were not built earlier. This will reliably cause dials.rlv to
    fail even thought the build setup was apparently fine.
    """
    import gltbx.fonts  # noqa: F401
    import gltbx.gl  # noqa: F401

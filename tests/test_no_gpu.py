# tests/test_no_gpu.py
import subprocess
import sys
import textwrap
import unittest


def _run_without_gpu_libs(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh subprocess where importing cupy/voltools
    (and their submodules) always raises ImportError, simulating a
    machine with no working GPU/CUDA install -- regardless of whether
    the host actually has one. This function does not check the return code
    and expects any tests using it to check the return code explicitly.
    This file was introduced to test PR #346"""
    setup = textwrap.dedent(
        """
        import builtins
        _real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "cupy" or name.startswith("cupy.") \\
               or name == "voltools" or name.startswith("voltools."):
                raise ImportError(f"simulated missing GPU: no module '{name}'")
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = _fake_import
        """
    )
    full_script = setup + "\n" + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", full_script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class TestEntryPointsWithoutGPU(unittest.TestCase):
    def test_merge_stars_importable_without_gpu(self):
        result = _run_without_gpu_libs(
            """
            from pytom_tm.entry_points import merge_stars
            merge_stars(['--help'])
            """
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_extract_candidates_importable_without_gpu(self):
        result = _run_without_gpu_libs(
            """
            from pytom_tm.entry_points import extract_candidates
            extract_candidates(['--help'])
            """
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

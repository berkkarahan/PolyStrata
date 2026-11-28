"""Tests for SETUP.bat: delegates to START-HERE.bat --setup, forwards args/exit code, and uses CRLF line endings."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STUB = REPO / "SETUP.bat"


class Text(unittest.TestCase):
    def test_wrapper_calls_start_here_setup_and_propagates_exit_code(self):
        data = STUB.read_bytes()
        text = data.decode("ascii")
        self.assertTrue(text.startswith("@echo off"))
        self.assertIn('call "%~dp0START-HERE.bat" --setup %*', text)
        self.assertIn("exit /b %errorlevel%", text)
        self.assertNotIn("\n\n", text.replace("\r\n", "\n"))

    def test_it_is_a_batch_file_with_cr_lf_line_ends(self):
        data = STUB.read_bytes()
        self.assertIn(b"\r\n", data)
        self.assertNotIn(b"\n", data.replace(b"\r\n", b""))
        self.assertIn("*.bat text eol=crlf", (REPO / ".gitattributes").read_text(encoding="utf-8"))


@unittest.skipUnless(sys.platform == "win32" and shutil.which("cmd"), "needs Windows cmd")
class Run(unittest.TestCase):
    def _run_stub(self, fake_start_here: bytes, args: list[str]):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            shutil.copy(STUB, folder / "SETUP.bat")
            (folder / "START-HERE.bat").write_bytes(fake_start_here)
            env = {k: v for k, v in os.environ.items() if k != "NoDefaultCurrentDirectoryInExePath"}
            return subprocess.run(
                ["cmd", "/c", str(folder / "SETUP.bat"), *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                cwd=d,
                env=env,
                timeout=60,
            )

    def test_passes_setup_flag_extra_args_and_exit_code(self):
        done = self._run_stub(b"@echo off\r\necho args=%*\r\nexit /b 3\r\n", ["--model", "IQ2_XS", "--yes"])
        self.assertEqual(done.returncode, 3)
        self.assertEqual(done.stderr.strip(), "")
        self.assertIn("args=--setup --model IQ2_XS --yes", done.stdout)

    def test_runs_with_no_extra_args(self):
        done = self._run_stub(b"@echo off\r\necho args=%*\r\nexit /b 0\r\n", [])
        self.assertEqual(done.returncode, 0)
        self.assertEqual(done.stderr.strip(), "")
        self.assertIn("args=--setup", done.stdout)


if __name__ == "__main__":
    unittest.main()

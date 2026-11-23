"""Optional pytest integration for `terminal/tests/e2e/`.

The project's `.venv` Python runs `unittest discover` without
`pytest` installed, while `tools/check --only py-tests` runs `pytest`.
Guarding `import pytest` keeps this file safely importable in both environments.
"""
from __future__ import annotations

import sys
from pathlib import Path

TERMINAL_ROOT = Path(__file__).resolve().parents[2]
if str(TERMINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TERMINAL_ROOT))

try:
    import pytest
except ImportError:
    pytest = None

from tests.e2e.harness import IsolatedWorkspace, WorkspaceSnapshotGuard  # noqa: E402


if pytest is not None:

    @pytest.fixture(scope="package", autouse=True)
    def e2e_repo_snapshot_guard():
        """Fail the E2E pytest package if any E2E test mutates the repository checkout."""
        guard = WorkspaceSnapshotGuard()
        yield guard
        guard.assert_unchanged("pytest e2e package")

    @pytest.fixture()
    def isolated_workspace():
        """Per-test isolated temporary workspace with full env and module patching."""
        with IsolatedWorkspace() as ws:
            yield ws

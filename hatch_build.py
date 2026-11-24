"""Hatch build hook that ensures the compiled web UI bundle is present.

The browser assets are bundled from TypeScript with esbuild. Building during
packaging (wheel/sdist) guarantees that PyPI artifacts ship ``static/dist``.
Editable installs (``uv sync``) also build when dependencies are already
installed, but degrade gracefully (warn and continue) when Node or
``node_modules`` is unavailable so developer setups and Node-less CI keep
working. Release builds fail loudly instead of publishing a broken UI.

Docker images build the bundle in a dedicated Node stage and copy it into the
source tree before packaging, so this hook is a no-op there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _run(command: list[str], cwd: Path) -> None:
    """Run *command* in *cwd*, handling Windows ``.cmd`` shims."""
    if os.name == "nt":
        subprocess.run(subprocess.list2cmdline(command), cwd=cwd, shell=True, check=True)
    else:
        subprocess.run(command, cwd=cwd, check=True)


class CustomBuildHook(BuildHookInterface):
    """Build the esbuild bundle before producing a wheel or sdist."""

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        bundle = root / "src/abstracts_explorer/web_ui/static/dist/app.js"

        if bundle.exists():
            return

        editable = version == "editable"

        npm = shutil.which("npm")
        node = shutil.which("node")
        if npm is None or node is None:
            if editable:
                self.app.display_warning(
                    "Web UI bundle is missing and Node.js/npm are unavailable; skipping the "
                    "frontend build for this editable install. Run `npm ci && npm run build:js` "
                    "to build it."
                )
                return
            raise RuntimeError(
                "The web UI bundle is missing and Node.js/npm are not available. "
                "Run `npm ci --ignore-scripts && npm run build:js` before building the package."
            )

        node_modules = root / "node_modules"
        if not node_modules.exists():
            if editable:
                self.app.display_warning(
                    "Web UI bundle is missing and node_modules is not installed; skipping the "
                    "frontend build for this editable install. Run `npm ci && npm run build:js` "
                    "to build it."
                )
                return
            _run([npm, "ci", "--ignore-scripts"], root)
            esbuild_install = node_modules / "esbuild" / "install.js"
            if esbuild_install.exists():
                _run([node, str(esbuild_install)], root)

        _run([npm, "run", "build:js"], root)

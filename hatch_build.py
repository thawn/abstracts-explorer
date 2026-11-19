"""Hatch build hook that ensures the compiled web UI bundle is present.

The browser assets are bundled from TypeScript with esbuild. Docker images
build the bundle in a dedicated Node stage and copy it into the source tree
before packaging, so this hook is a no-op there (Node is absent but the bundle
already exists). Local ``uv build`` runs the bundle build when Node is
available, and fails loudly when it is not and no bundle is present.

Editable installs (``uv sync``) are skipped: the package then runs straight from
the source tree, so requiring Node or a prebuilt bundle would break developer
setups (and Windows CI, where npm may not be on ``PATH``).
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
        # Editable installs use the source tree directly; no bundle is needed.
        if version == "editable":
            return

        root = Path(self.root)
        bundle = root / "src/abstracts_explorer/web_ui/static/dist/app.js"

        if bundle.exists():
            return

        npm = shutil.which("npm")
        node = shutil.which("node")
        if npm is None or node is None:
            raise RuntimeError(
                "The web UI bundle is missing and Node.js/npm are not available. "
                "Run `npm ci --ignore-scripts && npm run build:js` before building the package."
            )

        node_modules = root / "node_modules"
        if not node_modules.exists():
            _run([npm, "ci", "--ignore-scripts"], root)
            esbuild_install = node_modules / "esbuild" / "install.js"
            if esbuild_install.exists():
                _run([node, str(esbuild_install)], root)

        _run([npm, "run", "build:js"], root)

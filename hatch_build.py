"""Hatch build hook that ensures the compiled web UI bundle is present.

The browser assets are bundled from TypeScript with esbuild. Docker images
build the bundle in a dedicated Node stage and copy it into the source tree
before packaging, so this hook is a no-op there (Node is absent but the bundle
already exists). Local ``uv build`` runs the bundle build when Node is
available, and fails loudly when it is not and no bundle is present.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Build the esbuild bundle before producing a wheel or sdist."""

    def initialize(self, version: str, build_data: dict) -> None:
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
            subprocess.run(["npm", "ci", "--ignore-scripts"], cwd=root, check=True)
            esbuild_install = node_modules / "esbuild" / "install.js"
            if esbuild_install.exists():
                subprocess.run(["node", str(esbuild_install)], cwd=root, check=True)

        subprocess.run(["npm", "run", "build:js"], cwd=root, check=True)

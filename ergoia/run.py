"""Entrada rapida para ejecutar ErgoIA desde el boton Run de VS Code.

VS Code a veces ejecuta el archivo con el Python global de Windows aunque el
proyecto tenga un entorno virtual. Este launcher se relanza automaticamente con
`.venv\\Scripts\\python.exe` antes de importar dependencias como MediaPipe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def relaunch_with_venv() -> None:
    project_dir = Path(__file__).resolve().parent
    venv_python = project_dir / ".venv" / "Scripts" / "python.exe"

    if not venv_python.exists():
        raise RuntimeError(
            "No se encontro .venv\\Scripts\\python.exe. Crea el entorno con: "
            "python -m venv .venv && .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )

    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()

    if current_python != target_python:
        os.execv(str(target_python), [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    relaunch_with_venv()

    from infer import main

    main()

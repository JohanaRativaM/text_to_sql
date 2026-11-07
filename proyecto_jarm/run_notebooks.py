"""Ejecuta los notebooks 01-04 en orden y guarda las salidas (verificación de reproducibilidad).

Uso: python run_notebooks.py [01 02 03 04]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
NOTEBOOKS = {
    "01": "01_exploracion_datos.ipynb",
    "02": "02_consolidacion_datasets.ipynb",
    "03": "03_anotacion_schema_linking.ipynb",
    "04": "04_validacion_etiquetas.ipynb",
}


def run(name: str) -> None:
    path = NB_DIR / name
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(NB_DIR)}})
    t0 = time.time()
    client.execute()
    nbformat.write(nb, path)
    print(f"OK  {name}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    keys = sys.argv[1:] or list(NOTEBOOKS)
    for k in keys:
        run(NOTEBOOKS[k])

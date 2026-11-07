"""Carga y consolidación de las fuentes Spider, BIRD y Catastro.

Produce el dataset maestro homogéneo (``data/processed/master_dataset.parquet``)
y la partición por base de datos de BIRD (``data/processed/bird_db_split.json``).

Reglas metodológicas implementadas:
    * Spider conserva sus particiones originales (train -> train, dev -> validation).
    * BIRD reserva 14 bases completas del train original como prueba interna,
      seleccionadas de forma reproducible con la semilla 42; dev -> validation.
    * Catastro se asigna íntegramente a ``test_transfer`` (solo evaluación).
    * No se traducen preguntas ni se modifica el SQL.
    * Las evidencias vacías de BIRD se conservan.
    * Los duplicados se marcan, nunca se eliminan.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src import config
from src.config import get_logger

logger = get_logger(__name__, "data_loader.log")


# ---------------------------------------------------------------------------
# Utilidades genéricas
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    """Carga un archivo JSON en UTF-8 y lanza un error descriptivo si falla."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def relative_to_project(path: Path) -> str:
    """Devuelve la ruta como texto relativo a la raíz del proyecto (POSIX)."""
    try:
        return Path(path).resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def make_example_id(source: str, source_split: str, index: int) -> str:
    """Identificador único y reproducible: ``<SOURCE>-<split>-<índice>``."""
    return f"{source}-{source_split}-{index:05d}"


def _normalize_text(text: str) -> str:
    """Normalización ligera usada solo para detectar duplicados."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# ---------------------------------------------------------------------------
# Inventario de archivos originales
# ---------------------------------------------------------------------------
def inventory_raw_files() -> pd.DataFrame:
    """Inventario de los archivos presentes en ``raw/`` (sin abrir bases sqlite).

    Returns:
        DataFrame con ruta relativa, fuente, formato, tamaño y número de
        registros (cuando el archivo es JSON o Excel).
    """
    rows: list[dict[str, Any]] = []
    for source, folder in (
        ("SPIDER", config.SPIDER_RAW_DIR),
        ("BIRD", config.BIRD_RAW_DIR),
        ("CATASTRO", config.CATASTRO_RAW_DIR),
    ):
        if not folder.exists():
            rows.append({"source": source, "path": relative_to_project(folder),
                         "format": "MISSING", "size_mb": None, "n_records": None})
            continue
        for p in sorted(folder.iterdir()):
            if p.is_dir():
                n_sub = sum(1 for _ in p.iterdir())
                rows.append({"source": source, "path": relative_to_project(p),
                             "format": "directory", "size_mb": None, "n_records": n_sub})
                continue
            suffix = p.suffix.lower()
            n_records: int | None = None
            fmt = suffix.lstrip(".") or "no-extension"
            try:
                if suffix == ".json":
                    data = load_json(p)
                    n_records = len(data) if isinstance(data, list) else None
                elif suffix == ".xlsx":
                    n_records = len(pd.read_excel(p))
                elif suffix == ".sql":
                    with p.open("r", encoding="utf-8", errors="replace") as fh:
                        n_records = sum(1 for _ in fh)
                    fmt = "sql (líneas)"
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudo leer %s: %s", p, exc)
            rows.append({"source": source, "path": relative_to_project(p), "format": fmt,
                         "size_mb": round(p.stat().st_size / 1e6, 2), "n_records": n_records})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spider
# ---------------------------------------------------------------------------
def load_spider_raw(include_extra: bool = False) -> pd.DataFrame:
    """Carga los ejemplos de Spider (train_spider + dev).

    Args:
        include_extra: si es ``True`` incluye también ``train_others.json`` y
            ``test.json`` (solo con fines exploratorios; la propuesta define
            7.000 train / 1.034 dev).
    """
    parts: list[tuple[str, Path]] = [
        ("train", config.SPIDER_FILES["train"]),
        ("dev", config.SPIDER_FILES["dev"]),
    ]
    if include_extra:
        parts += [("train_others", config.SPIDER_FILES["train_others"]),
                  ("test", config.SPIDER_FILES["test"])]
    frames = []
    for split, path in parts:
        data = load_json(path)
        df = pd.DataFrame(
            {
                "source_split": split,
                "source_file": relative_to_project(path),
                "source_index": range(len(data)),
                "db_id": [d["db_id"] for d in data],
                "question": [d["question"] for d in data],
                "sql_gold": [d["query"] for d in data],
            }
        )
        logger.info("Spider %s: %d ejemplos, %d bases", split, len(df), df.db_id.nunique())
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# BIRD
# ---------------------------------------------------------------------------
def load_bird_raw() -> pd.DataFrame:
    """Carga los ejemplos de BIRD (train + dev) conservando ``evidence``."""
    frames = []
    for split in ("train", "dev"):
        path = config.BIRD_FILES[split]
        data = load_json(path)
        df = pd.DataFrame(
            {
                "source_split": split,
                "source_file": relative_to_project(path),
                "source_index": range(len(data)),
                "db_id": [d["db_id"] for d in data],
                "question": [d["question"] for d in data],
                "sql_gold": [d["SQL"] for d in data],
                "evidence": [d.get("evidence", "") if d.get("evidence") is not None else "" for d in data],
                "difficulty": [d.get("difficulty") for d in data],
            }
        )
        logger.info("BIRD %s: %d ejemplos, %d bases, %d evidencias vacías",
                    split, len(df), df.db_id.nunique(), (df.evidence.str.strip() == "").sum())
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_bird_db_split(
    train_db_ids: Iterable[str],
    seed: int = config.RANDOM_SEED,
    n_test_dbs: int = config.BIRD_N_TEST_DBS,
) -> dict[str, list[str]]:
    """Selecciona de forma reproducible las bases de BIRD reservadas como prueba.

    Procedimiento: ordenar alfabéticamente los ``db_id`` del train original,
    permutarlos con ``numpy.random.default_rng(seed)`` y tomar los primeros
    ``n_test_dbs``. Con seed=42 y n=14 se obtienen exactamente 1.432 ejemplos,
    coincidiendo con la partición definida en la propuesta.
    """
    ordered = sorted(set(train_db_ids))
    rng = np.random.default_rng(seed)
    permuted = [ordered[i] for i in rng.permutation(len(ordered))]
    test_dbs = sorted(permuted[:n_test_dbs])
    train_dbs = sorted(set(ordered) - set(test_dbs))
    return {"seed": seed, "n_test_dbs": n_test_dbs, "train_dbs": train_dbs, "test_dbs": test_dbs}


def get_or_create_bird_split(train_db_ids: Iterable[str], path: Path = config.BIRD_SPLIT_PATH) -> dict:
    """Reutiliza la partición persistida si existe; si no, la construye y guarda."""
    if path.exists():
        split = load_json(path)
        logger.info("Partición BIRD cargada desde %s", relative_to_project(path))
        rebuilt = build_bird_db_split(train_db_ids, split.get("seed", config.RANDOM_SEED),
                                      split.get("n_test_dbs", config.BIRD_N_TEST_DBS))
        if rebuilt["test_dbs"] != split["test_dbs"]:
            logger.warning("La partición persistida difiere de la reconstruida con la semilla; "
                           "se conserva la persistida para no alterar experimentos previos.")
        return split
    split = build_bird_db_split(train_db_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2, ensure_ascii=False)
    logger.info("Partición BIRD creada y guardada en %s", relative_to_project(path))
    return split


# ---------------------------------------------------------------------------
# Catastro
# ---------------------------------------------------------------------------
def _detect_catastro_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    """Detecta las columnas de pregunta y SQL del Excel por su contenido/nombre."""
    cols = list(df.columns)
    sql_col = next((c for c in cols if "sql" in str(c).lower()), None)
    q_col = next((c for c in cols if any(k in str(c).lower() for k in ("regla", "pregunta", "question"))), None)
    id_col = next((c for c in cols if str(c).strip().lower().startswith("n")
                   and c not in (sql_col, q_col)), None)
    if sql_col is None or q_col is None:
        raise ValueError(f"No se pudieron identificar columnas de pregunta/SQL en el Excel. Columnas: {cols}")
    return q_col, sql_col, id_col


def load_catastro_raw() -> pd.DataFrame:
    """Carga los 80 pares regla-SQL del Excel catastral adaptándose a sus columnas."""
    path = config.CATASTRO_FILES["questions"]
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el Excel catastral: {path}")
    raw = pd.read_excel(path)
    q_col, sql_col, id_col = _detect_catastro_columns(raw)
    df = pd.DataFrame(
        {
            "source_split": "catastro",
            "source_file": relative_to_project(path),
            "source_index": range(len(raw)),
            "source_row_number": raw[id_col].tolist() if id_col else range(1, len(raw) + 1),
            "db_id": config.CATASTRO_DB_ID,
            "question": raw[q_col].astype(str).str.strip(),
            "sql_gold": raw[sql_col].astype(str).str.strip(),
        }
    )
    logger.info("Catastro: %d pares (columnas detectadas: pregunta='%s', sql='%s')", len(df), q_col, sql_col)
    return df


# ---------------------------------------------------------------------------
# Dataset maestro
# ---------------------------------------------------------------------------
def flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Marca duplicados exactos de (source, db_id, pregunta, SQL) sin eliminarlos."""
    key = (
        df["source"].astype(str) + "|" + df["db_id"].astype(str) + "|"
        + df["question"].map(_normalize_text) + "|" + df["sql_gold"].map(_normalize_text)
    )
    group_id = key.map(lambda k: hashlib.md5(k.encode("utf-8")).hexdigest()[:10])
    dup_mask = key.duplicated(keep=False)
    df = df.copy()
    df["is_duplicate_question_sql"] = dup_mask
    df["duplicate_group_id"] = np.where(dup_mask, group_id, None)
    logger.info("Duplicados exactos pregunta+SQL marcados: %d registros en %d grupos",
                int(dup_mask.sum()), df.loc[dup_mask, "duplicate_group_id"].nunique())
    return df


def build_master_dataset(save: bool = True) -> pd.DataFrame:
    """Construye el dataset maestro homogéneo con Spider, BIRD y Catastro.

    Args:
        save: si es ``True`` persiste ``master_dataset.parquet``.

    Returns:
        DataFrame con las columnas de :data:`src.config.MASTER_COLUMNS`.
    """
    config.ensure_directories()
    frames: list[pd.DataFrame] = []

    # --- Spider -----------------------------------------------------------
    spider = load_spider_raw()
    spider["source"] = "SPIDER"
    spider["split"] = spider["source_split"].map({"train": "train", "dev": "validation"})
    spider["evidence"] = None
    spider["difficulty"] = None
    spider["schema_reference"] = relative_to_project(config.SPIDER_FILES["tables"]) + "#" + spider["db_id"]
    frames.append(spider)

    # --- BIRD -------------------------------------------------------------
    bird = load_bird_raw()
    bird["source"] = "BIRD"
    train_dbs = bird.loc[bird.source_split == "train", "db_id"].unique()
    split_info = get_or_create_bird_split(train_dbs)
    test_dbs = set(split_info["test_dbs"])

    def _bird_split(row: pd.Series) -> str:
        if row.source_split == "dev":
            return "validation"
        return "test" if row.db_id in test_dbs else "train"

    bird["split"] = bird.apply(_bird_split, axis=1)
    bird["schema_reference"] = np.where(
        bird.source_split == "train",
        relative_to_project(config.BIRD_FILES["train_tables"]) + "#" + bird["db_id"],
        relative_to_project(config.BIRD_FILES["dev_tables"]) + "#" + bird["db_id"],
    )
    frames.append(bird)

    # --- Catastro ---------------------------------------------------------
    try:
        cat = load_catastro_raw()
        cat["source"] = "CATASTRO"
        cat["split"] = "test_transfer"
        cat["evidence"] = None
        cat["difficulty"] = None
        cat["schema_reference"] = (relative_to_project(config.CATASTRO_FILES["dump_sql"])
                                   + "#" + config.CATASTRO_PG_SCHEMA)
        frames.append(cat)
    except FileNotFoundError as exc:
        logger.error("Catastro no disponible: %s", exc)

    master = pd.concat(frames, ignore_index=True)
    master["example_id"] = [
        make_example_id(s, sp, i) for s, sp, i in zip(master.source, master.source_split, master.source_index)
    ]
    master["dialect"] = master["source"].map(config.DIALECTS)
    master["schema_id"] = master["source"] + ":" + master["db_id"]
    master = flag_duplicates(master)

    extra_cols = [c for c in master.columns if c not in config.MASTER_COLUMNS]
    master = master[config.MASTER_COLUMNS + extra_cols]

    if master.example_id.duplicated().any():
        raise RuntimeError("Se generaron example_id duplicados; revisar la lógica de identificadores.")

    logger.info("Dataset maestro: %d registros", len(master))
    logger.info("Distribución por fuente/split:\n%s", master.groupby(["source", "split"]).size().to_string())

    if save:
        master.to_parquet(config.MASTER_DATASET_PATH, index=False)
        logger.info("Guardado %s", relative_to_project(config.MASTER_DATASET_PATH))
    return master


def load_master_dataset(path: Path = config.MASTER_DATASET_PATH) -> pd.DataFrame:
    """Carga el dataset maestro desde Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute build_master_dataset() primero.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    build_master_dataset(save=True)

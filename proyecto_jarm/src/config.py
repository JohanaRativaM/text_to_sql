"""Configuración centralizada del proyecto.

Todas las rutas son relativas a la raíz del proyecto (carpeta que contiene
``src/``), de modo que el código es portable entre equipos.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Los datos originales se encuentran en ``raw/`` (no se mueven a data/raw).
RAW_DIR: Path = PROJECT_ROOT / "raw"
SPIDER_RAW_DIR: Path = RAW_DIR / "spider_data"
BIRD_RAW_DIR: Path = RAW_DIR / "bird_data"
CATASTRO_RAW_DIR: Path = RAW_DIR / "catastro_data"

DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ANNOTATIONS_DIR: Path = DATA_DIR / "annotations"
SCHEMA_LINKING_DIR: Path = DATA_DIR / "schema_linking"
TEXT_TO_SQL_DIR: Path = DATA_DIR / "text_to_sql"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
STATISTICS_DIR: Path = OUTPUTS_DIR / "statistics"
QUALITY_DIR: Path = OUTPUTS_DIR / "quality_reports"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

# Archivos de salida principales
MASTER_DATASET_PATH: Path = PROCESSED_DIR / "master_dataset.parquet"
SCHEMAS_PATH: Path = PROCESSED_DIR / "schemas.json"
SQL_ANALYSIS_PATH: Path = PROCESSED_DIR / "sql_analysis.parquet"
BIRD_SPLIT_PATH: Path = PROCESSED_DIR / "bird_db_split.json"

ANNOTATION_PATHS: dict[str, Path] = {
    "SPIDER": ANNOTATIONS_DIR / "spider_annotations.parquet",
    "BIRD": ANNOTATIONS_DIR / "bird_annotations.parquet",
    "CATASTRO": ANNOTATIONS_DIR / "catastro_annotations.parquet",
}

# Archivos originales (verificados en la inspección inicial)
SPIDER_FILES: dict[str, Path] = {
    "train": SPIDER_RAW_DIR / "train_spider.json",
    "dev": SPIDER_RAW_DIR / "dev.json",
    "tables": SPIDER_RAW_DIR / "tables.json",
    # Disponibles pero fuera del alcance definido en la propuesta (7.000 / 1.034)
    "train_others": SPIDER_RAW_DIR / "train_others.json",
    "test": SPIDER_RAW_DIR / "test.json",
    "test_tables": SPIDER_RAW_DIR / "test_tables.json",
}
BIRD_FILES: dict[str, Path] = {
    "train": BIRD_RAW_DIR / "train.json",
    "dev": BIRD_RAW_DIR / "dev.json",
    "train_tables": BIRD_RAW_DIR / "train_tables.json",
    "dev_tables": BIRD_RAW_DIR / "dev_tables.json",
}
CATASTRO_FILES: dict[str, Path] = {
    "questions": CATASTRO_RAW_DIR / "catastral_SQL.xlsx",
    "dump_sql": CATASTRO_RAW_DIR / "bd_catastro_test.sql",
}

# ---------------------------------------------------------------------------
# Constantes metodológicas
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

SOURCES: tuple[str, ...] = ("SPIDER", "BIRD", "CATASTRO")

# Partición BIRD: se reservan 14 bases completas del train original como
# prueba interna. Procedimiento reproducible verificado en la inspección:
# np.random.default_rng(42).permutation(sorted(db_ids))[:14] -> 1.432 ejemplos.
BIRD_N_TEST_DBS: int = 14

DIALECTS: dict[str, str] = {
    "SPIDER": "sqlite",
    "BIRD": "sqlite",       # Las bases BIRD se distribuyen como .sqlite
    "CATASTRO": "postgres",
}

CATASTRO_DB_ID: str = "catastro_unidad_intervencion_18"
CATASTRO_PG_SCHEMA: str = "unidad_intervencion_18"

ROLE_LABELS: list[str] = [
    "SELECT",
    "WHERE",
    "JOIN",
    "FROM",
    "GROUP_BY",
    "HAVING",
    "ORDER_BY",
]

ANNOTATION_STATUS: tuple[str, ...] = (
    "automatic",
    "manual_review",
    "validated",
    "unresolved",
)

# Columnas del dataset maestro (orden canónico)
MASTER_COLUMNS: list[str] = [
    "example_id",
    "source",
    "db_id",
    "split",
    "source_split",
    "source_file",
    "source_index",
    "question",
    "sql_gold",
    "evidence",
    "dialect",
    "schema_id",
    "schema_reference",
    "difficulty",
    "is_duplicate_question_sql",
    "duplicate_group_id",
]


def ensure_directories() -> None:
    """Crea las carpetas de salida si no existen (nunca toca ``raw/``)."""
    for d in (
        PROCESSED_DIR,
        ANNOTATIONS_DIR,
        SCHEMA_LINKING_DIR,
        TEXT_TO_SQL_DIR,
        STATISTICS_DIR,
        QUALITY_DIR,
        LOGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Devuelve un logger con salida a consola y, opcionalmente, a archivo.

    Args:
        name: nombre del logger (normalmente ``__name__``).
        log_file: nombre del archivo dentro de ``outputs/logs``; si es ``None``
            solo se registra en consola.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger

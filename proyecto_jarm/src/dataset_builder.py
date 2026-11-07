"""Construcción de los corpus supervisados a partir de las anotaciones maestras.

Schema Linking (esta entrega):
    * ``relevance_{train,validation,test}.jsonl``  : clasificación binaria.
    * ``roles_{train,validation,test}.jsonl``      : clasificación multietiqueta (7 roles).
    * ``relevance_test_transfer.jsonl`` / ``roles_test_transfer.jsonl`` : Catastro,
      **solo evaluación** (se genera en archivos separados, nunca en train/validation/test).

Text-to-SQL (siguiente entrega): se deja definida la función
:func:`build_generator_example` y se genera una muestra de formato.

Decisiones de diseño:
    * El SQL gold NO se incluye como entrada; solo se usó para derivar etiquetas.
    * El esquema completo se referencia por ``schema_id``: su serialización
      compacta y sus claves foráneas se escriben una sola vez en
      ``schema_linking/schemas_compact.json`` (el catálogo detallado está en
      ``processed/schemas.json``). Cada registro incluye además el contexto local
      de la tabla del elemento (``element_context``). Incluir el esquema completo
      en cada línea (``inline_schema=True``) multiplica el tamaño (>5 GB en train
      y >11 GB en Catastro), por lo que está desactivado por defecto.
    * Por defecto solo se exportan anotaciones con estado ``automatic`` o
      ``validated``; las ``manual_review``/``unresolved`` quedan excluidas y
      contabilizadas en el manifiesto.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src import config
from src.config import ROLE_LABELS, get_logger
from src.data_loader import relative_to_project
from src.schema_annotator import ROLE_COLUMNS

logger = get_logger(__name__, "dataset_builder.log")

TRAINABLE_SOURCES = ("SPIDER", "BIRD")
DEFAULT_ALLOWED_STATUS = ("automatic", "validated")
GENERATOR_INSTRUCTION = ("Generate a valid SQL query for the following question using the provided "
                         "database schema.")


# ---------------------------------------------------------------------------
# Serialización de esquemas
# ---------------------------------------------------------------------------
def serialize_schema_compact(schema: dict) -> str:
    """Serialización compacta: ``table(col:type, ...) | table2(...)``."""
    cols_by_table: dict[int, list[str]] = {}
    for c in schema["columns"]:
        t = c["column_type"] or ""
        cols_by_table.setdefault(c["table_index"], []).append(f"{c['name_original']}:{t}" if t else c["name_original"])
    parts = [f"{t['name_original']}({', '.join(cols_by_table.get(t['table_index'], []))})" for t in schema["tables"]]
    return " | ".join(parts)


def table_context(schema: dict, table_name: str) -> str:
    """Definición compacta de una sola tabla (contexto local del elemento)."""
    cols = [c["name_original"] for c in schema["columns"] if c["table_name"] == table_name]
    return f"{table_name}({', '.join(cols)})"


def _schema_texts(catalog: dict[str, dict], schema_id: str, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if schema_id not in cache:
        schema = catalog[schema_id]
        cache[schema_id] = {
            "compact": serialize_schema_compact(schema),
            "tables": {t["name_original"]: table_context(schema, t["name_original"]) for t in schema["tables"]},
            "fks": [f"{fk['from_table']}.{fk['from_column']}={fk['to_table']}.{fk['to_column']}"
                    for fk in schema["foreign_keys"]],
        }
    return cache[schema_id]


# ---------------------------------------------------------------------------
# Schema Linking
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_compact_schemas(catalog: dict[str, dict], schema_ids: Iterable[str],
                          path: Path | None = None) -> Path:
    """Escribe ``schemas_compact.json`` (schema_id -> serialización compacta + FKs)."""
    path = path or (config.SCHEMA_LINKING_DIR / "schemas_compact.json")
    cache: dict[str, dict[str, Any]] = {}
    payload = {}
    for sid in sorted(set(schema_ids)):
        texts = _schema_texts(catalog, sid, cache)
        payload[sid] = {"schema": texts["compact"], "foreign_keys": texts["fks"], "tables": texts["tables"]}
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    logger.info("Esquemas compactos (%d) guardados en %s", len(payload), relative_to_project(path))
    return path


def annotations_to_records(ann: pd.DataFrame, master_idx: pd.DataFrame, catalog: dict[str, dict],
                           task: str, inline_schema: bool = False) -> Iterable[dict]:
    """Transforma anotaciones en registros JSONL de la tarea indicada.

    Args:
        ann: anotaciones ya filtradas (fuente, split, estado, relevancia).
        master_idx: dataset maestro indexado por ``example_id`` (para evidence/schema_id).
        task: ``"schema_relevance"`` o ``"schema_roles"``.
        inline_schema: incluir la serialización compacta del esquema completo.
    """
    cache: dict[str, dict[str, Any]] = {}
    for r in ann.itertuples(index=False):
        m = master_idx.loc[r.example_id]
        texts = _schema_texts(catalog, m.schema_id, cache)
        evidence = m.evidence if isinstance(m.evidence, str) and m.evidence.strip() else None
        rec: dict[str, Any] = {
            "task": task,
            "example_id": r.example_id,
            "source": r.source,
            "db_id": r.db_id,
            "schema_id": m.schema_id,
            "split": r.split,
            "question": r.question,
            "evidence": evidence,
            "schema": texts["compact"] if inline_schema else None,
            "schema_foreign_keys": texts["fks"] if inline_schema else None,
            "schema_ref": None if inline_schema else "schema_linking/schemas_compact.json",
            "element": r.element_id,
            "element_type": r.element_type,
            "element_context": texts["tables"].get(r.table_name),
            "annotation_status": r.annotation_status,
        }
        if task == "schema_relevance":
            rec["label"] = int(r.relevance)
        else:
            rec["roles"] = [int(x) for x in r.roles]
            rec["role_labels"] = ROLE_LABELS
        yield rec


def build_schema_linking_datasets(master: pd.DataFrame, catalog: dict[str, dict],
                                  annotations_by_source: dict[str, pd.DataFrame],
                                  allowed_status: Iterable[str] = DEFAULT_ALLOWED_STATUS,
                                  inline_schema: bool = False, include_transfer: bool = True) -> dict[str, Any]:
    """Genera los JSONL de relevancia y roles por partición.

    Catastro nunca se incorpora a train/validation/test; si ``include_transfer``
    es ``True`` se escribe en archivos ``*_test_transfer.jsonl`` separados para la
    evaluación de transferencia.
    """
    config.ensure_directories()
    allowed = set(allowed_status)
    master_idx = master.set_index("example_id")
    used_schema_ids: set[str] = set()
    for df in annotations_by_source.values():
        used_schema_ids.update(master_idx.loc[df.example_id.unique(), "schema_id"].unique().tolist())
    compact_path = write_compact_schemas(catalog, used_schema_ids)
    manifest: dict[str, Any] = {"allowed_status": sorted(allowed), "inline_schema": inline_schema,
                                "schema_ref": relative_to_project(compact_path),
                                "role_labels": ROLE_LABELS, "files": {}, "excluded": {}}

    train_frames = [annotations_by_source[s] for s in TRAINABLE_SOURCES if s in annotations_by_source]
    trainable = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    jobs: list[tuple[str, pd.DataFrame]] = [(sp, trainable[trainable.split == sp]) for sp in ("train", "validation", "test")]
    if include_transfer and "CATASTRO" in annotations_by_source:
        jobs.append(("test_transfer", annotations_by_source["CATASTRO"]))

    for split, df in jobs:
        if df.empty:
            continue
        assert split == "test_transfer" or not (df.source == "CATASTRO").any(), "Catastro no puede entrar al entrenamiento"
        keep = df[df.annotation_status.isin(allowed)]
        manifest["excluded"][split] = {
            "annotations_excluded_by_status": int(len(df) - len(keep)),
            "questions_excluded_by_status": int(df.loc[~df.annotation_status.isin(allowed), "example_id"].nunique()),
        }
        # Relevancia: positivos y negativos
        p = config.SCHEMA_LINKING_DIR / f"relevance_{split}.jsonl"
        n = _write_jsonl(p, annotations_to_records(keep, master_idx, catalog, "schema_relevance", inline_schema))
        manifest["files"][p.name] = {"n_records": n, "n_positive": int(keep.relevance.sum()),
                                     "n_negative": int((keep.relevance == 0).sum()),
                                     "n_questions": int(keep.example_id.nunique()), "n_dbs": int(keep.db_id.nunique()),
                                     "sources": sorted(keep.source.unique().tolist())}
        logger.info("%s: %d registros (%d pos / %d neg)", p.name, n, manifest["files"][p.name]["n_positive"],
                    manifest["files"][p.name]["n_negative"])
        # Roles: solo relevantes
        rel = keep[keep.relevance == 1]
        p = config.SCHEMA_LINKING_DIR / f"roles_{split}.jsonl"
        n = _write_jsonl(p, annotations_to_records(rel, master_idx, catalog, "schema_roles", inline_schema))
        manifest["files"][p.name] = {"n_records": n, "n_questions": int(rel.example_id.nunique()),
                                     "n_dbs": int(rel.db_id.nunique()),
                                     "role_frequency": {r: int(rel[c].sum()) for r, c in zip(ROLE_LABELS, ROLE_COLUMNS)},
                                     "sources": sorted(rel.source.unique().tolist())}
        logger.info("%s: %d registros", p.name, n)

    with (config.SCHEMA_LINKING_DIR / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info("Manifiesto guardado en %s", relative_to_project(config.SCHEMA_LINKING_DIR / "manifest.json"))
    return manifest


# ---------------------------------------------------------------------------
# Text-to-SQL (estructura para la siguiente entrega)
# ---------------------------------------------------------------------------
def build_generator_example(example: pd.Series, annotations: pd.DataFrame, schema: dict,
                            use_evidence: bool = True) -> dict[str, Any]:
    """Construye un ejemplo del generador con esquema enfocado derivado de anotaciones gold.

    La entrada no contiene el SQL gold; el esquema enfocado se limita a las
    tablas relevantes (con todas sus columnas relevantes marcadas) y los roles
    provienen de las anotaciones.
    """
    rel = annotations[(annotations.example_id == example.example_id) & (annotations.relevance == 1)]
    rel_tables = set(rel[rel.element_type == "table"].element_id)
    rel_columns = rel[rel.element_type == "column"]
    focused = []
    for t in schema["tables"]:
        if t["name_original"] not in rel_tables:
            continue
        cols = [c["name_original"] for c in schema["columns"] if c["table_name"] == t["name_original"]]
        focused.append(f"{t['name_original']}({', '.join(cols)})")
    roles = [{"element": r.element_id, "element_type": r.element_type,
              "roles": [lab for lab, v in zip(ROLE_LABELS, r.roles) if v]} for r in rel.itertuples()]
    evidence = example.evidence if use_evidence and isinstance(example.evidence, str) and example.evidence.strip() else None
    return {
        "instruction": GENERATOR_INSTRUCTION,
        "example_id": example.example_id, "source": example.source, "db_id": example.db_id,
        "split": example.split, "dialect": example.dialect,
        "input": {
            "question": example.question,
            "schema": " | ".join(focused),
            "relevant_elements": rel.element_id.tolist(),
            "roles": roles,
            "evidence": evidence,
        },
        "output": example.sql_gold,
    }


def write_generator_format_sample(master: pd.DataFrame, catalog: dict[str, dict],
                                  annotations_by_source: dict[str, pd.DataFrame], n_per_split: int = 5) -> dict[str, int]:
    """Escribe una muestra del formato del generador (``generator_format_sample_<split>.jsonl``).

    La construcción definitiva del corpus generativo se realizará en la
    siguiente entrega, una vez validadas las anotaciones. Catastro no se incluye.
    """
    config.ensure_directories()
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        sub = master[(master.split == split) & (master.source.isin(TRAINABLE_SOURCES))].head(n_per_split)
        records = []
        for ex in sub.itertuples(index=False):
            ann = annotations_by_source[ex.source]
            records.append(build_generator_example(pd.Series(ex._asdict()), ann, catalog[ex.schema_id]))
        p = config.TEXT_TO_SQL_DIR / f"generator_format_sample_{split}.jsonl"
        counts[p.name] = _write_jsonl(p, records)
    spec = {
        "status": "formato definido; construcción completa pendiente para la siguiente entrega",
        "instruction": GENERATOR_INSTRUCTION,
        "input_fields": ["question", "schema (enfocado, derivado de anotaciones gold)", "relevant_elements",
                         "roles", "evidence (BIRD, opcional)"],
        "output": "sql_gold",
        "constraints": ["El SQL gold no se usa para construir la entrada",
                        "Catastro no se incorpora al entrenamiento generativo",
                        "Particiones por base de datos idénticas a schema_linking"],
    }
    with (config.TEXT_TO_SQL_DIR / "generator_format_spec.json").open("w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    logger.info("Muestras de formato del generador: %s", counts)
    return counts


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Lee un archivo JSONL (opcionalmente solo las primeras ``limit`` líneas)."""
    out = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            out.append(json.loads(line))
    return out


if __name__ == "__main__":
    from src.data_loader import load_master_dataset
    from src.schema_annotator import load_annotations
    from src.schema_parser import load_schema_catalog

    master_df = load_master_dataset()
    cat = load_schema_catalog()
    anns = {s: load_annotations(s) for s in config.SOURCES if config.ANNOTATION_PATHS[s].exists()}
    build_schema_linking_datasets(master_df, cat, anns)
    write_generator_format_sample(master_df, cat, anns)

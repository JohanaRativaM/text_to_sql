"""Construcción de etiquetas de Schema Linking (relevancia + roles).

Unidad de anotación: (pregunta, elemento candidato del esquema).
Para cada pregunta se generan candidatos para **todas** las tablas y columnas
de su esquema (positivos y negativos).

Etiquetas:
    relevance : 1 si el elemento participa en el SQL gold (según el parser), 0 si no.
    roles     : vector binario de 7 posiciones en el orden de ``config.ROLE_LABELS``
                [SELECT, WHERE, JOIN, FROM, GROUP_BY, HAVING, ORDER_BY].

Estados de anotación (``annotation_status``):
    automatic     : la consulta se resolvió completamente; etiqueta confiable.
    manual_review : la consulta tiene referencias no resueltas (tabla/columna
                    inexistente en el catálogo, alias no resuelto...). Los
                    positivos detectados son correctos, pero los negativos podrían
                    ser falsos negativos; requiere revisión.
    unresolved    : el elemento es candidato de una referencia ambigua que no
                    pudo desambiguarse, o la consulta no pudo parsearse.
    validated     : reservado para revisión humana posterior (no se asigna aquí).

Las etiquetas se derivan exclusivamente del SQL gold; nunca de coincidencias
textuales entre el nombre del elemento y la pregunta.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src import config
from src.config import ROLE_LABELS, get_logger
from src.data_loader import relative_to_project

logger = get_logger(__name__, "schema_annotator.log")

ROLE_COLUMNS = [f"role_{r}" for r in ROLE_LABELS]
ANNOTATION_COLUMNS = [
    "example_id", "source", "db_id", "split", "question", "sql_gold",
    "element_id", "element_name", "element_type", "table_name",
    "relevance", "roles", *ROLE_COLUMNS,
    "annotation_status", "parser_status", "annotation_notes",
]


def roles_vector(roles: Iterable[str]) -> list[int]:
    """Convierte un conjunto de roles en el vector binario canónico de 7 posiciones."""
    rs = set(roles)
    return [1 if r in rs else 0 for r in ROLE_LABELS]


def roles_dict(vector: list[int]) -> dict[str, int]:
    """Representación ``{"SELECT": 1, ...}`` de un vector de roles."""
    return dict(zip(ROLE_LABELS, vector))


def _ambiguous_candidates(details: dict) -> dict[str, list[str]]:
    """Elementos candidatos de referencias ambiguas -> lista de referencias crudas."""
    out: dict[str, list[str]] = {}
    for u in details.get("unresolved", []):
        if u.get("reason") == "ambiguous":
            for cand in u.get("candidates", []):
                out.setdefault(cand, []).append(u["raw"])
    return out


def annotate_example(example: dict[str, Any], analysis: dict[str, Any], schema: dict) -> list[dict[str, Any]]:
    """Genera las anotaciones (una fila por elemento del esquema) para una pregunta.

    Args:
        example: fila del dataset maestro (dict).
        analysis: fila de ``sql_analysis`` (dict) con ``analysis_json`` ya decodificado
            en la clave ``details``.
        schema: esquema del catálogo.
    """
    parse_ok = analysis.get("parse_status") == "ok"
    parser_status = analysis.get("resolution_status", "parse_error")
    details = analysis.get("details") or {}
    table_roles: dict[str, list[str]] = details.get("table_roles", {}) if parse_ok else {}
    column_roles: dict[str, list[str]] = details.get("column_roles", {}) if parse_ok else {}
    unresolved = details.get("unresolved", []) if parse_ok else []
    ambiguous = _ambiguous_candidates(details) if parse_ok else {}
    star_tables = sorted({t for s in details.get("stars", []) for t in s.get("tables", [])}) if parse_ok else []

    if not parse_ok:
        base_status = "unresolved"
        base_note = f"parse_error: {analysis.get('parse_error')}"
    elif unresolved:
        base_status = "manual_review"
        reasons = sorted({f"{u['kind']}:{u['raw']}:{u['reason']}" for u in unresolved})
        base_note = "unresolved_references=" + "; ".join(reasons)[:500]
    else:
        base_status = "automatic"
        base_note = None

    common = {
        "example_id": example["example_id"], "source": example["source"], "db_id": example["db_id"],
        "split": example["split"], "question": example["question"], "sql_gold": example["sql_gold"],
        "parser_status": parser_status,
    }
    rows: list[dict[str, Any]] = []

    def make_row(element_id: str, element_name: str, element_type: str, table_name: str | None,
                 roles: list[str] | None) -> dict[str, Any]:
        relevant = roles is not None
        vec = roles_vector(roles or [])
        status, note = base_status, base_note
        notes: list[str] = [note] if note else []
        if element_id in ambiguous:
            status = "unresolved"
            notes.append("ambiguous_reference=" + ",".join(ambiguous[element_id]))
        if relevant and sum(vec) == 0:
            notes.append("relevant_without_role_clause")
        if element_type == "table" and element_id in star_tables and relevant:
            notes.append("star_projection")
        row = {**common, "element_id": element_id, "element_name": element_name, "element_type": element_type,
               "table_name": table_name, "relevance": 1 if relevant else 0, "roles": vec,
               "annotation_status": status, "annotation_notes": "; ".join(notes) if notes else None}
        row.update(dict(zip(ROLE_COLUMNS, vec)))
        return row

    for t in schema["tables"]:
        eid = t["element_id"]
        rows.append(make_row(eid, t["name_original"], "table", t["name_original"], table_roles.get(eid)))
    for c in schema["columns"]:
        eid = c["element_id"]
        rows.append(make_row(eid, c["name_original"], "column", c["table_name"], column_roles.get(eid)))
    return rows


def build_annotations(master: pd.DataFrame, analysis: pd.DataFrame, catalog: dict[str, dict],
                      sources: Iterable[str] | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    """Genera los archivos de anotaciones por fuente.

    Para evitar cargar todo en memoria, se procesa una fuente a la vez y se
    guarda su Parquet antes de pasar a la siguiente (guardado incremental).

    Returns:
        Diccionario ``source -> DataFrame de anotaciones``.
    """
    config.ensure_directories()
    sources = tuple(sources) if sources else config.SOURCES
    analysis_idx = analysis.set_index("example_id")
    outputs: dict[str, pd.DataFrame] = {}

    for source in sources:
        subset = master[master.source == source]
        if subset.empty:
            logger.warning("Fuente %s sin registros; se omite", source)
            continue
        rows: list[dict[str, Any]] = []
        missing_analysis = 0
        for rec in subset.to_dict("records"):
            if rec["example_id"] not in analysis_idx.index:
                missing_analysis += 1
                continue
            a = analysis_idx.loc[rec["example_id"]].to_dict()
            a["details"] = json.loads(a["analysis_json"]) if isinstance(a.get("analysis_json"), str) else {}
            schema = catalog.get(rec["schema_id"])
            if schema is None:
                logger.error("Esquema %s no encontrado para %s", rec["schema_id"], rec["example_id"])
                continue
            rows.extend(annotate_example(rec, a, schema))
        df = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
        df["relevance"] = df["relevance"].astype("int8")
        for c in ROLE_COLUMNS:
            df[c] = df[c].astype("int8")
        outputs[source] = df
        logger.info("%s: %d anotaciones (%d preguntas, %d sin análisis) | relevancia=1: %d | estados: %s",
                    source, len(df), df.example_id.nunique(), missing_analysis, int(df.relevance.sum()),
                    df.annotation_status.value_counts().to_dict())
        if save:
            path = config.ANNOTATION_PATHS[source]
            df.to_parquet(path, index=False)
            logger.info("Guardado %s", relative_to_project(path))
    return outputs


def load_annotations(source: str, path: Path | None = None) -> pd.DataFrame:
    """Carga las anotaciones de una fuente."""
    path = path or config.ANNOTATION_PATHS[source]
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute build_annotations() primero.")
    return pd.read_parquet(path)


def load_all_annotations(sources: Iterable[str] | None = None) -> pd.DataFrame:
    """Concatena las anotaciones de varias fuentes (por defecto todas las existentes)."""
    sources = tuple(sources) if sources else config.SOURCES
    frames = [load_annotations(s) for s in sources if config.ANNOTATION_PATHS[s].exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ANNOTATION_COLUMNS)


def example_annotation_view(annotations: pd.DataFrame, example_id: str, only_relevant: bool = True) -> pd.DataFrame:
    """Vista legible de las anotaciones de una pregunta (para los notebooks)."""
    sub = annotations[annotations.example_id == example_id]
    if only_relevant:
        sub = sub[sub.relevance == 1]
    cols = ["element_id", "element_type", "relevance", *ROLE_COLUMNS, "annotation_status", "annotation_notes"]
    return sub[cols].reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import load_master_dataset
    from src.schema_parser import load_schema_catalog
    from src.sql_parser import load_sql_analysis

    build_annotations(load_master_dataset(), load_sql_analysis(), load_schema_catalog(), save=True)

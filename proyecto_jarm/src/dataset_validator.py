"""Controles automáticos de calidad para el dataset maestro, los esquemas,
el análisis SQL, las anotaciones y las particiones.

Ningún control elimina registros: los problemas se documentan en
``outputs/quality_reports/`` para revisión.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src import config
from src.config import ROLE_LABELS, get_logger
from src.data_loader import relative_to_project
from src.schema_annotator import ROLE_COLUMNS

logger = get_logger(__name__, "dataset_validator.log")


class ValidationReport:
    """Acumula resultados de validación (resumen + detalles por control)."""

    def __init__(self) -> None:
        self.summary: list[dict[str, Any]] = []
        self.details: dict[str, pd.DataFrame] = {}

    def add(self, category: str, check: str, source: str, count: int, severity: str = "warning",
            note: str | None = None) -> None:
        self.summary.append({"category": category, "check": check, "source": source, "count": int(count),
                             "severity": severity if count > 0 else "ok", "note": note})

    def add_detail(self, name: str, df: pd.DataFrame) -> None:
        if df is not None and len(df) > 0:
            self.details[name] = pd.concat([self.details[name], df], ignore_index=True) if name in self.details else df

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.summary)

    def save(self, prefix: str = "") -> dict[str, str]:
        config.ensure_directories()
        paths: dict[str, str] = {}
        p = config.QUALITY_DIR / f"{prefix}validation_summary.csv"
        self.summary_frame().to_csv(p, index=False, encoding="utf-8")
        paths["summary"] = relative_to_project(p)
        for name, df in self.details.items():
            p = config.QUALITY_DIR / f"{prefix}{name}.csv"
            df.to_csv(p, index=False, encoding="utf-8")
            paths[name] = relative_to_project(p)
        logger.info("Reportes de calidad guardados: %s", list(paths.values()))
        return paths


# ---------------------------------------------------------------------------
# Integridad del dataset maestro
# ---------------------------------------------------------------------------
def validate_master(master: pd.DataFrame, catalog: dict[str, dict], report: ValidationReport) -> None:
    """Controles de integridad sobre ``master_dataset``."""
    for source, df in master.groupby("source"):
        empty_q = df.question.isna() | (df.question.astype(str).str.strip() == "")
        empty_sql = df.sql_gold.isna() | (df.sql_gold.astype(str).str.strip() == "")
        report.add("integridad", "preguntas_vacias", source, empty_q.sum(), "error")
        report.add("integridad", "preguntas_sin_sql", source, empty_sql.sum(), "error")
        no_schema = ~df.schema_id.isin(catalog.keys())
        report.add("integridad", "sql_sin_esquema_asociado", source, no_schema.sum(), "error")
        report.add("integridad", "example_id_duplicados", source, df.example_id.duplicated().sum(), "error")
        report.add("integridad", "registros_duplicados_pregunta_sql", source, int(df.is_duplicate_question_sql.sum()),
                   "warning", "marcados, no eliminados")
        for col in ("db_id", "split", "source_split", "dialect", "schema_id"):
            report.add("integridad", f"nulos_en_{col}", source, df[col].isna().sum(), "error")
        if source == "BIRD":
            report.add("integridad", "evidence_nula", source, df.evidence.isna().sum(), "error",
                       "BIRD debe conservar evidence (cadena vacía permitida)")
            report.add("integridad", "evidence_vacia", source, (df.evidence.fillna("").str.strip() == "").sum(),
                       "info", "conservadas por diseño")
        dups = df[df.is_duplicate_question_sql].sort_values("duplicate_group_id")
        if len(dups):
            report.add_detail("duplicates", dups[["example_id", "source", "db_id", "split", "duplicate_group_id",
                                                  "question", "sql_gold"]])


# ---------------------------------------------------------------------------
# Consistencia de esquema (según el análisis SQL)
# ---------------------------------------------------------------------------
def validate_schema_consistency(analysis: pd.DataFrame, catalog: dict[str, dict], report: ValidationReport) -> None:
    """Tablas/columnas inexistentes, alias sin resolver, referencias ambiguas y FKs."""
    rows: list[dict[str, Any]] = []
    for rec in analysis.itertuples(index=False):
        if rec.parse_status != "ok":
            rows.append({"example_id": rec.example_id, "source": rec.source, "db_id": rec.db_id,
                         "kind": "parse", "raw": None, "reason": rec.parse_error, "clause": None, "candidates": None})
            continue
        details = json.loads(rec.analysis_json)
        for u in details.get("unresolved", []):
            rows.append({"example_id": rec.example_id, "source": rec.source, "db_id": rec.db_id, "kind": u["kind"],
                         "raw": u["raw"], "reason": u["reason"], "clause": u.get("clause"),
                         "candidates": ",".join(u.get("candidates", []))})
    unresolved = pd.DataFrame(rows, columns=["example_id", "source", "db_id", "kind", "raw", "reason", "clause", "candidates"])
    report.add_detail("unresolved_references", unresolved)

    for source, df in analysis.groupby("source"):
        u = unresolved[unresolved.source == source]
        report.add("parser", "consultas_no_parseadas", source, (df.parse_status != "ok").sum(), "error")
        report.add("parser", "consultas_resueltas", source, (df.resolution_status == "resolved").sum(), "info")
        report.add("parser", "consultas_parcialmente_resueltas", source, (df.resolution_status == "partial").sum())
        report.add("parser", "consultas_sin_resolver", source, (df.resolution_status == "unresolved").sum())
        report.add("esquema", "tablas_usadas_inexistentes", source,
                   ((u.kind == "table") & (u.reason.str.startswith("table_not_in_schema"))).sum())
        report.add("esquema", "columnas_usadas_inexistentes", source,
                   ((u.kind == "column") & (u.reason.str.startswith(("column_not_in_table", "column_not_found")))).sum())
        report.add("esquema", "alias_sin_resolver", source, ((u.kind == "column") & (u.reason == "alias_not_found")).sum())
        report.add("esquema", "referencias_ambiguas", source, (u.reason == "ambiguous").sum())
        report.add("esquema", "alias_duplicados_en_sql", source, (u.kind == "scope").sum())
        report.add("parser", "literales_sqlite_entre_comillas_dobles", source, int(df.n_literal_fallbacks.sum()), "info",
                   "interpretados como literal (semántica SQLite), no como error")

    # Claves foráneas y relaciones del catálogo
    fk_rows = []
    for sid, s in catalog.items():
        col_set = {c["element_id"].lower() for c in s["columns"]}
        bad = [fk for fk in s["foreign_keys"]
               if f"{fk['from_table']}.{fk['from_column']}".lower() not in col_set
               or f"{fk['to_table']}.{fk['to_column']}".lower() not in col_set]
        fk_rows.append({"schema_id": sid, "source": s["source"], "n_fk": len(s["foreign_keys"]), "n_fk_invalid": len(bad),
                        "n_issues": len(s["issues"]), "issues": "; ".join(s["issues"])[:300]})
    fk_df = pd.DataFrame(fk_rows)
    for source, df in fk_df.groupby("source"):
        report.add("esquema", "fk_con_referencias_invalidas", source, int(df.n_fk_invalid.sum()), "error")
        report.add("esquema", "esquemas_con_incidencias", source, int((df.n_issues > 0).sum()))
        report.add("esquema", "esquemas_sin_fk", source, int((df.n_fk == 0).sum()), "info")
    report.add_detail("schema_issues", fk_df[(fk_df.n_fk_invalid > 0) | (fk_df.n_issues > 0)])


# ---------------------------------------------------------------------------
# Consistencia de etiquetas
# ---------------------------------------------------------------------------
_ROLE_REQUIRES = {"role_WHERE": "has_where", "role_GROUP_BY": "has_group_by", "role_HAVING": "has_having",
                  "role_ORDER_BY": "has_order_by"}


def validate_annotations(annotations: pd.DataFrame, analysis: pd.DataFrame, catalog: dict[str, dict],
                         report: ValidationReport) -> None:
    """Controles de consistencia de etiquetas de relevancia y roles."""
    source = annotations.source.iloc[0] if len(annotations) else "?"
    problems: list[pd.DataFrame] = []

    # Vectores de roles de dimensión 7 y coherentes con las columnas role_*
    lens = annotations.roles.map(len)
    report.add("etiquetas", "vectores_roles_dimension_distinta_7", source, (lens != 7).sum(), "error")
    vec_sum = annotations.roles.map(sum)
    report.add("etiquetas", "vector_roles_incoherente_con_columnas", source,
               (vec_sum != annotations[ROLE_COLUMNS].sum(axis=1)).sum(), "error")

    # Irrelevantes con roles positivos
    irr_roles = annotations[(annotations.relevance == 0) & (vec_sum > 0)]
    report.add("etiquetas", "irrelevantes_con_roles", source, len(irr_roles), "error")
    problems.append(irr_roles.assign(check="irrelevantes_con_roles"))

    # Relevantes sin roles (válido solo si se documenta el motivo)
    rel_no_roles = annotations[(annotations.relevance == 1) & (vec_sum == 0)]
    report.add("etiquetas", "relevantes_sin_roles", source, len(rel_no_roles), "warning",
               "columnas usadas fuera de las 7 cláusulas (p.ej. LIMIT) o tablas sin FROM")
    problems.append(rel_no_roles.assign(check="relevantes_sin_roles"))

    # Elementos duplicados dentro de la misma pregunta
    dup = annotations[annotations.duplicated(["example_id", "element_id"], keep=False)]
    report.add("etiquetas", "elementos_duplicados_por_pregunta", source, len(dup), "error")
    problems.append(dup.assign(check="elementos_duplicados_por_pregunta"))

    # Elementos utilizados (según el parser) con relevancia 0 y cobertura de candidatos
    ann_idx = annotations.set_index(["example_id", "element_id"]).relevance
    a_src = analysis[analysis.source == source]
    used_missing = 0
    used_rows = []
    cand_mismatch = []
    cand_counts = annotations.groupby("example_id").size()
    schema_sizes = {sid: s["n_tables"] + s["n_columns"] for sid, s in catalog.items()}
    for rec in a_src.itertuples(index=False):
        if rec.parse_status != "ok":
            continue
        d = json.loads(rec.analysis_json)
        used = list(d.get("table_roles", {})) + list(d.get("column_roles", {}))
        for eid in used:
            val = ann_idx.get((rec.example_id, eid))
            if val is None or val != 1:
                used_missing += 1
                used_rows.append({"example_id": rec.example_id, "element_id": eid, "check": "usado_con_relevancia_0"})
        expected = schema_sizes.get(f"{rec.source}:{rec.db_id}")
        if expected is not None and cand_counts.get(rec.example_id, 0) != expected:
            cand_mismatch.append({"example_id": rec.example_id, "expected": expected,
                                  "found": int(cand_counts.get(rec.example_id, 0))})
    report.add("etiquetas", "elementos_usados_con_relevancia_0", source, used_missing, "error")
    report.add("etiquetas", "preguntas_con_candidatos_incompletos", source, len(cand_mismatch), "error")
    if used_rows:
        problems.append(pd.DataFrame(used_rows))
    if cand_mismatch:
        report.add_detail("candidate_coverage_mismatch", pd.DataFrame(cand_mismatch).assign(source=source))

    # Roles inconsistentes con las cláusulas presentes en el SQL
    feats = a_src.set_index("example_id")[["has_where", "has_group_by", "has_having", "has_order_by", "has_join"]]
    merged = annotations[annotations.relevance == 1].join(feats, on="example_id")
    incons = pd.Series(False, index=merged.index)
    for role_col, feat in _ROLE_REQUIRES.items():
        incons |= (merged[role_col] == 1) & (~merged[feat].fillna(False).astype(bool))
    incons |= (merged["role_JOIN"] == 1) & (~merged["has_join"].fillna(False).astype(bool))
    report.add("etiquetas", "roles_inconsistentes_con_clausulas_sql", source, int(incons.sum()), "error")
    problems.append(merged[incons].assign(check="roles_inconsistentes_con_clausulas_sql"))

    # Estados
    for status, n in annotations.annotation_status.value_counts().items():
        report.add("etiquetas", f"anotaciones_{status}", source, n, "info")
    q_review = annotations[annotations.annotation_status != "automatic"].drop_duplicates("example_id")
    report.add("etiquetas", "preguntas_pendientes_revision_manual", source, len(q_review), "warning")
    if len(q_review):
        report.add_detail("questions_manual_review",
                          q_review[["example_id", "source", "db_id", "split", "parser_status", "annotation_status",
                                    "question", "sql_gold", "annotation_notes"]])

    cols = ["example_id", "source", "element_id", "element_type", "relevance", *ROLE_COLUMNS, "check"]
    prob = pd.concat([p for p in problems if len(p)], ignore_index=True) if any(len(p) for p in problems) else None
    if prob is not None:
        for c in cols:
            if c not in prob.columns:
                prob[c] = None
        report.add_detail("label_inconsistencies", prob[cols])


# ---------------------------------------------------------------------------
# Particiones
# ---------------------------------------------------------------------------
def validate_partitions(master: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Distribución por fuente/split, bases por partición e intersecciones de db_id."""
    dist = master.groupby(["source", "split"]).agg(n_examples=("example_id", "size"), n_dbs=("db_id", "nunique")).reset_index()
    report.add_detail("partition_distribution", dist)
    for source, df in master.groupby("source"):
        splits = {s: set(g.db_id) for s, g in df.groupby("split")}
        names = sorted(splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                inter = splits[a] & splits[b]
                report.add("particiones", f"db_id_compartidos_{a}_vs_{b}", source, len(inter), "error",
                           ",".join(sorted(inter))[:200] if inter else None)
    cat_train = master[(master.source == "CATASTRO") & (master.split != "test_transfer")]
    report.add("particiones", "catastro_fuera_de_test_transfer", "CATASTRO", len(cat_train), "error")
    expected = {("SPIDER", "train"): 7000, ("SPIDER", "validation"): 1034, ("BIRD", "train"): 7996,
                ("BIRD", "validation"): 1534, ("BIRD", "test"): 1432, ("CATASTRO", "test_transfer"): 80}
    for (src, sp), n in expected.items():
        found = int(dist[(dist.source == src) & (dist.split == sp)].n_examples.sum())
        report.add("particiones", f"tamano_esperado_{sp}", src, abs(found - n), "error",
                   f"esperado={n}, encontrado={found}")
    return dist


def validate_schema_linking_files(report: ValidationReport) -> None:
    """Verifica que Catastro no aparezca en los corpus JSONL de clasificación."""
    for task in ("relevance", "roles"):
        for split in ("train", "validation", "test"):
            p = config.SCHEMA_LINKING_DIR / f"{task}_{split}.jsonl"
            if not p.exists():
                continue
            n_cat = 0
            n = 0
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    n += 1
                    if '"source": "CATASTRO"' in line or "CATASTRO-" in line[:60]:
                        n_cat += 1
            report.add("particiones", f"catastro_en_{task}_{split}", "SCHEMA_LINKING", n_cat, "error", f"total_lineas={n}")


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def run_all_validations(master: pd.DataFrame, catalog: dict[str, dict], analysis: pd.DataFrame,
                        annotations_by_source: dict[str, pd.DataFrame], save: bool = True) -> ValidationReport:
    """Ejecuta todos los controles y guarda los reportes en ``outputs/quality_reports``."""
    report = ValidationReport()
    validate_master(master, catalog, report)
    validate_schema_consistency(analysis, catalog, report)
    for source, ann in annotations_by_source.items():
        validate_annotations(ann, analysis, catalog, report)
    validate_partitions(master, report)
    validate_schema_linking_files(report)
    summary = report.summary_frame()
    errors = summary[(summary.severity == "error")]
    logger.info("Validación: %d controles, %d con errores, %d advertencias", len(summary), len(errors),
                (summary.severity == "warning").sum())
    if len(errors):
        logger.warning("Controles con errores:\n%s", errors[["category", "check", "source", "count", "note"]].to_string())
    if save:
        report.save()
    return report


if __name__ == "__main__":
    from src.data_loader import load_master_dataset
    from src.schema_annotator import load_annotations
    from src.schema_parser import load_schema_catalog
    from src.sql_parser import load_sql_analysis

    anns = {s: load_annotations(s) for s in config.SOURCES if config.ANNOTATION_PATHS[s].exists()}
    run_all_validations(load_master_dataset(), load_schema_catalog(), load_sql_analysis(), anns)

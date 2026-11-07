"""Analítica descriptiva: tablas CSV y gráficos en ``outputs/statistics``.

Cada función devuelve el DataFrame calculado y, si ``save=True``, lo persiste.
Los gráficos se generan con matplotlib/seaborn y se guardan como PNG.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.config import ROLE_LABELS, get_logger  # noqa: E402
from src.schema_annotator import ROLE_COLUMNS  # noqa: E402

logger = get_logger(__name__, "statistics.log")
sns.set_theme(style="whitegrid")

SQL_FEATURES = ["has_join", "has_subquery", "has_cte", "has_where", "has_group_by", "has_having",
                "has_order_by", "has_limit", "has_distinct", "has_aggregation", "has_set_operation", "has_window"]


def _save(df: pd.DataFrame, name: str, save: bool) -> pd.DataFrame:
    if save:
        config.ensure_directories()
        df.to_csv(config.STATISTICS_DIR / f"{name}.csv", index=False, encoding="utf-8")
    return df


def _savefig(fig: plt.Figure, name: str, save: bool) -> None:
    if save:
        fig.tight_layout()
        fig.savefig(config.STATISTICS_DIR / f"{name}.png", dpi=120)
    plt.close(fig)


# 1. Preguntas por fuente y partición -----------------------------------------
def questions_by_source_split(master: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    df = (master.groupby(["source", "split", "source_split"])
          .agg(n_questions=("example_id", "size"), n_dbs=("db_id", "nunique"),
               n_duplicates=("is_duplicate_question_sql", "sum")).reset_index())
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="source", y="n_questions", hue="split", ax=ax)
    ax.set_title("Preguntas por fuente y partición")
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    _savefig(fig, "01_questions_by_source_split", save)
    return _save(df, "01_questions_by_source_split", save)


# 2-4. Esquemas, tablas y columnas por esquema ---------------------------------
def schema_statistics(schema_summary: pd.DataFrame, master: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    used = set(master.schema_id)
    ss = schema_summary.assign(used_in_master=schema_summary.schema_id.isin(used))
    df = (ss.groupby("source")
          .agg(n_schemas=("schema_id", "size"), n_schemas_used=("used_in_master", "sum"),
               tables_mean=("n_tables", "mean"), tables_median=("n_tables", "median"), tables_min=("n_tables", "min"),
               tables_max=("n_tables", "max"), columns_mean=("n_columns", "mean"), columns_median=("n_columns", "median"),
               columns_min=("n_columns", "min"), columns_max=("n_columns", "max"),
               fk_mean=("n_foreign_keys", "mean")).reset_index().round(2))
    bench = ss[ss.source != "CATASTRO"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=bench, x="n_tables", hue="source", bins=25, ax=axes[0], element="step")
    axes[0].set_title("Tablas por esquema (Spider / BIRD)")
    sns.histplot(data=bench, x="n_columns", hue="source", bins=40, ax=axes[1], element="step")
    axes[1].set_title("Columnas por esquema (Spider / BIRD)")
    _savefig(fig, "03_tables_columns_per_schema", save)
    _save(ss, "02_schema_summary_detail", save)
    return _save(df, "02_schema_statistics", save)


# 5. Candidatos por pregunta ----------------------------------------------------
def candidates_per_question(annotations: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    per_q = (annotations.groupby(["source", "example_id", "element_type"]).size().unstack(fill_value=0)
             .reset_index())
    per_q["total"] = per_q.get("table", 0) + per_q.get("column", 0)
    df = per_q.groupby("source").agg(n_questions=("example_id", "size"), tables_mean=("table", "mean"),
                                     columns_mean=("column", "mean"), total_mean=("total", "mean"),
                                     total_median=("total", "median"), total_min=("total", "min"),
                                     total_max=("total", "max")).reset_index().round(2)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=per_q[per_q.source != "CATASTRO"], x="source", y="total", ax=ax)
    ax.set_title("Candidatos (tablas + columnas) por pregunta")
    _savefig(fig, "05_candidates_per_question", save)
    return _save(df, "05_candidates_per_question", save)


# 6. Relevancia -----------------------------------------------------------------
def relevance_distribution(annotations: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    df = (annotations.groupby(["source", "split", "element_type", "relevance"]).size()
          .rename("n").reset_index())
    tot = df.groupby(["source", "split", "element_type"]).n.transform("sum")
    df["pct"] = (df.n / tot * 100).round(2)
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = df[df.relevance == 1].copy()
    pos["grp"] = pos.source + "/" + pos.split
    sns.barplot(data=pos, x="grp", y="pct", hue="element_type", ax=ax, errorbar=None)
    ax.set_title("% de candidatos relevantes por fuente/partición")
    ax.set_ylabel("% relevancia = 1")
    ax.tick_params(axis="x", rotation=30)
    _savefig(fig, "06_relevance_distribution", save)
    return _save(df, "06_relevance_distribution", save)


# 7. Roles ----------------------------------------------------------------------
def roles_distribution(annotations: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    rel = annotations[annotations.relevance == 1]
    rows = []
    for (source, etype), g in rel.groupby(["source", "element_type"]):
        for role, col in zip(ROLE_LABELS, ROLE_COLUMNS):
            rows.append({"source": source, "element_type": etype, "role": role, "n": int(g[col].sum()),
                         "pct_of_relevant": round(g[col].mean() * 100, 2)})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, etype in zip(axes, ("table", "column")):
        sns.barplot(data=df[df.element_type == etype], x="role", y="pct_of_relevant", hue="source", ax=ax,
                    order=ROLE_LABELS, errorbar=None)
        ax.set_title(f"Roles entre {etype}s relevantes (% de relevantes)")
        ax.tick_params(axis="x", rotation=30)
    _savefig(fig, "07_roles_distribution", save)
    # multi-rol
    n_roles = rel[ROLE_COLUMNS].sum(axis=1)
    multi = rel.assign(n_roles=n_roles).groupby(["source", "n_roles"]).size().rename("n").reset_index()
    _save(multi, "07_roles_per_element_count", save)
    return _save(df, "07_roles_distribution", save)


# 8. Construcciones SQL ---------------------------------------------------------
def sql_constructs(analysis: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    ok = analysis[analysis.parse_status == "ok"]
    df = ok.groupby("source")[SQL_FEATURES].sum().astype(int).reset_index()
    pct = ok.groupby("source")[SQL_FEATURES].mean().mul(100).round(2).reset_index()
    long = df.melt(id_vars="source", var_name="construct", value_name="n")
    long["construct"] = long.construct.str.replace("has_", "")
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(data=pct.melt(id_vars="source", var_name="construct", value_name="pct")
                .assign(construct=lambda d: d.construct.str.replace("has_", "")),
                x="construct", y="pct", hue="source", ax=ax, errorbar=None)
    ax.set_title("% de consultas por construcción SQL")
    ax.tick_params(axis="x", rotation=30)
    _savefig(fig, "08_sql_constructs", save)
    _save(pct, "08_sql_constructs_pct", save)
    # Agregaciones
    agg_rows = []
    for src, g in ok.groupby("source"):
        counts: dict[str, int] = {}
        for a in g.aggregations:
            for name in json.loads(a):
                counts[name] = counts.get(name, 0) + 1
        agg_rows += [{"source": src, "aggregation": k, "n": v} for k, v in counts.items()]
    _save(pd.DataFrame(agg_rows), "08_aggregations", save)
    return _save(long, "08_sql_constructs", save)


# 9-12. Estado de anotación / parser --------------------------------------------
def annotation_status_statistics(annotations: pd.DataFrame, analysis: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    rows = []
    for source, g in annotations.groupby("source"):
        a = analysis[analysis.source == source]
        unresolved_cols = int(a.n_unresolved_columns.fillna(0).sum())
        unresolved_tabs = int(a.n_unresolved_tables.fillna(0).sum())
        rows.append({
            "source": source,
            "n_annotations": len(g),
            "n_automatic": int((g.annotation_status == "automatic").sum()),
            "n_manual_review": int((g.annotation_status == "manual_review").sum()),
            "n_unresolved": int((g.annotation_status == "unresolved").sum()),
            "n_validated": int((g.annotation_status == "validated").sum()),
            "questions_total": int(g.example_id.nunique()),
            "questions_manual_review": int(g.loc[g.annotation_status != "automatic", "example_id"].nunique()),
            "queries_not_parsed": int((a.parse_status != "ok").sum()),
            "queries_partially_resolved": int((a.resolution_status == "partial").sum()),
            "queries_unresolved": int((a.resolution_status == "unresolved").sum()),
            "unresolved_column_refs": unresolved_cols,
            "unresolved_table_refs": unresolved_tabs,
            "literal_fallbacks": int(a.n_literal_fallbacks.fillna(0).sum()),
        })
    df = pd.DataFrame(rows)
    return _save(df, "09_annotation_status", save)


def run_all_statistics(master: pd.DataFrame, schema_summary: pd.DataFrame, analysis: pd.DataFrame,
                       annotations: pd.DataFrame, save: bool = True) -> dict[str, pd.DataFrame]:
    """Genera todas las tablas y gráficos de analítica descriptiva."""
    out = {
        "questions": questions_by_source_split(master, save),
        "schemas": schema_statistics(schema_summary, master, save),
        "candidates": candidates_per_question(annotations, save),
        "relevance": relevance_distribution(annotations, save),
        "roles": roles_distribution(annotations, save),
        "sql_constructs": sql_constructs(analysis, save),
        "annotation_status": annotation_status_statistics(annotations, analysis, save),
    }
    logger.info("Estadísticas generadas en %s", config.STATISTICS_DIR)
    return out


if __name__ == "__main__":
    from src.data_loader import load_master_dataset
    from src.schema_annotator import load_all_annotations
    from src.schema_parser import load_schema_catalog, schema_summary as _schema_summary
    from src.sql_parser import load_sql_analysis

    run_all_statistics(load_master_dataset(), _schema_summary(load_schema_catalog()), load_sql_analysis(),
                       load_all_annotations())

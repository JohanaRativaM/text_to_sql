"""Catálogo homogéneo de esquemas para Spider, BIRD y Catastro.

Representación de cada esquema (``schemas.json``)::

    {
      "schema_id": "SPIDER:concert_singer",
      "db_id": "concert_singer", "source": "SPIDER", "dialect": "sqlite",
      "pg_schema": null,                      # solo Catastro
      "tables":  [{"table_index": 0, "name_original": "stadium", "name_normalized": "stadium",
                   "element_id": "stadium", "description": null, "n_columns": 6}],
      "columns": [{"column_index": 1, "table_index": 0, "table_name": "stadium",
                   "name_original": "Stadium_ID", "name_normalized": "stadium id",
                   "column_type": "number", "element_id": "stadium.Stadium_ID",
                   "is_primary_key": true, "description": null}],
      "primary_keys": [{"table": "stadium", "columns": ["Stadium_ID"]}],
      "foreign_keys": [{"from_table": ..., "from_column": ..., "to_table": ..., "to_column": ...}],
      "relations": [{"table_a": ..., "table_b": ..., "via": "stadium.x -> y.z"}],
      "ddl": "CREATE TABLE ...",
      "ddl_origin": "reconstructed_from_metadata" | "pg_dump",
      "n_tables": 2, "n_columns": 15,
      "issues": []
    }

Reglas:
    * El índice de tabla ``-1`` (columna ``*``) no es una columna real.
    * Cada columna queda asociada a una única tabla mediante ``table_index``.
    * Se conservan nombres originales y normalizados cuando existen ambos.
    * Para Catastro se usa el DDL real del ``pg_dump``; no se infieren relaciones.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import sqlglot
from sqlglot import exp

from src import config
from src.config import get_logger
from src.data_loader import load_json, relative_to_project

logger = get_logger(__name__, "schema_parser.log")

TableDict = dict[str, Any]
ColumnDict = dict[str, Any]
SchemaDict = dict[str, Any]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _quote(identifier: str) -> str:
    """Entrecomilla un identificador para el DDL (doble comilla SQL estándar)."""
    return '"' + str(identifier).replace('"', '""') + '"'


def _flatten_pk_indices(primary_keys: list) -> list[int]:
    """Aplana la lista de PKs (BIRD usa listas anidadas para claves compuestas)."""
    out: list[int] = []
    for pk in primary_keys:
        if isinstance(pk, (list, tuple)):
            out.extend(int(x) for x in pk)
        else:
            out.append(int(pk))
    return out


def build_ddl(schema: SchemaDict) -> str:
    """Reconstruye un DDL ``CREATE TABLE`` a partir de los metadatos del esquema.

    Solo usa información presente en el catálogo (nombres, tipos, PK y FK);
    """
    cols_by_table: dict[int, list[ColumnDict]] = {}
    for c in schema["columns"]:
        cols_by_table.setdefault(c["table_index"], []).append(c)
    pk_by_table = {pk["table"]: pk["columns"] for pk in schema["primary_keys"]}
    fks_by_table: dict[str, list[dict]] = {}
    for fk in schema["foreign_keys"]:
        fks_by_table.setdefault(fk["from_table"], []).append(fk)

    statements = []
    for t in schema["tables"]:
        lines = [f"  {_quote(c['name_original'])} {c['column_type'] or ''}".rstrip()
                 for c in cols_by_table.get(t["table_index"], [])]
        if t["name_original"] in pk_by_table:
            lines.append("  PRIMARY KEY (" + ", ".join(_quote(c) for c in pk_by_table[t["name_original"]]) + ")")
        for fk in fks_by_table.get(t["name_original"], []):
            lines.append(f"  FOREIGN KEY ({_quote(fk['from_column'])}) REFERENCES "
                         f"{_quote(fk['to_table'])}({_quote(fk['to_column'])})")
        statements.append(f"CREATE TABLE {_quote(t['name_original'])} (\n" + ",\n".join(lines) + "\n);")
    return "\n".join(statements)


def _build_relations(foreign_keys: list[dict]) -> list[dict]:
    """Relaciones entre tablas derivadas exclusivamente de las FK declaradas."""
    rels: dict[tuple[str, str], list[str]] = {}
    for fk in foreign_keys:
        key = (fk["from_table"], fk["to_table"])
        rels.setdefault(key, []).append(
            f"{fk['from_table']}.{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}")
    return [{"table_a": a, "table_b": b, "via": v} for (a, b), v in rels.items()]


# ---------------------------------------------------------------------------
# Spider / BIRD (formato tables.json)
# ---------------------------------------------------------------------------
def parse_benchmark_schema(entry: dict, source: str) -> SchemaDict:
    """Convierte una entrada de ``tables.json`` (Spider/BIRD) al formato homogéneo.

    Args:
        entry: diccionario con ``db_id``, ``table_names[_original]``,
            ``column_names[_original]``, ``column_types``, ``primary_keys``,
            ``foreign_keys``.
        source: ``"SPIDER"`` o ``"BIRD"``.
    """
    issues: list[str] = []
    db_id = entry["db_id"]
    t_orig = entry["table_names_original"]
    t_norm = entry.get("table_names", t_orig)
    c_orig = entry["column_names_original"]
    c_norm = entry.get("column_names", c_orig)
    c_types = entry.get("column_types", [None] * len(c_orig))

    if len(c_orig) != len(c_norm) or len(c_orig) != len(c_types):
        issues.append(f"Longitudes inconsistentes: column_names_original={len(c_orig)}, "
                      f"column_names={len(c_norm)}, column_types={len(c_types)}")
    if len(t_orig) != len(t_norm):
        issues.append(f"Longitudes inconsistentes en tablas: original={len(t_orig)}, normalized={len(t_norm)}")

    tables: list[TableDict] = [
        {"table_index": i, "name_original": t_orig[i],
         "name_normalized": t_norm[i] if i < len(t_norm) else t_orig[i],
         "element_id": t_orig[i], "description": None, "n_columns": 0}
        for i in range(len(t_orig))
    ]
    pk_indices = set(_flatten_pk_indices(entry.get("primary_keys", [])))

    columns: list[ColumnDict] = []
    for idx, (t_idx, name) in enumerate(c_orig):
        if t_idx == -1:
            continue  # comodín '*': no es una columna real
        if t_idx >= len(t_orig):
            issues.append(f"Columna {idx} '{name}' referencia tabla inexistente {t_idx}")
            continue
        columns.append({
            "column_index": idx,
            "table_index": t_idx,
            "table_name": t_orig[t_idx],
            "name_original": name,
            "name_normalized": c_norm[idx][1] if idx < len(c_norm) else name,
            "column_type": c_types[idx] if idx < len(c_types) else None,
            "element_id": f"{t_orig[t_idx]}.{name}",
            "is_primary_key": idx in pk_indices,
            "description": None,
        })
        tables[t_idx]["n_columns"] += 1

    col_by_idx = {c["column_index"]: c for c in columns}

    # Claves primarias (agrupadas por tabla, soportando claves compuestas)
    pk_map: dict[str, list[str]] = {}
    for pk in entry.get("primary_keys", []):
        idxs = pk if isinstance(pk, (list, tuple)) else [pk]
        for i in idxs:
            c = col_by_idx.get(int(i))
            if c is None:
                issues.append(f"primary_key índice {i} no corresponde a una columna real")
                continue
            pk_map.setdefault(c["table_name"], []).append(c["name_original"])
    primary_keys = [{"table": t, "columns": cols} for t, cols in pk_map.items()]

    foreign_keys: list[dict] = []
    for fk in entry.get("foreign_keys", []):
        try:
            a, b = col_by_idx.get(int(fk[0])), col_by_idx.get(int(fk[1]))
        except (TypeError, ValueError, IndexError):
            issues.append(f"foreign_key mal formada: {fk}")
            continue
        if a is None or b is None:
            issues.append(f"foreign_key con índices no resolubles: {fk}")
            continue
        foreign_keys.append({"from_table": a["table_name"], "from_column": a["name_original"],
                             "to_table": b["table_name"], "to_column": b["name_original"]})

    # Nombres de tabla duplicados (ignorando mayúsculas) dificultan la resolución
    lowered = [t.lower() for t in t_orig]
    if len(set(lowered)) != len(lowered):
        issues.append("Nombres de tabla duplicados (case-insensitive)")

    schema: SchemaDict = {
        "schema_id": f"{source}:{db_id}",
        "db_id": db_id,
        "source": source,
        "dialect": config.DIALECTS[source],
        "pg_schema": None,
        "tables": tables,
        "columns": columns,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "relations": _build_relations(foreign_keys),
        "ddl": None,
        "ddl_origin": "reconstructed_from_metadata",
        "n_tables": len(tables),
        "n_columns": len(columns),
        "issues": issues,
    }
    schema["ddl"] = build_ddl(schema)
    return schema


def iter_benchmark_schemas(path: Path, source: str) -> Iterator[SchemaDict]:
    """Itera perezosamente sobre los esquemas de un archivo ``tables.json``."""
    for entry in load_json(path):
        yield parse_benchmark_schema(entry, source)


# ---------------------------------------------------------------------------
# Catastro (pg_dump PostgreSQL)
# ---------------------------------------------------------------------------
_RE_CREATE_TABLE = re.compile(r"^(CREATE TABLE (\S+) \(.*?^\));", re.M | re.S)
_RE_PK = re.compile(r"^ALTER TABLE ONLY (\S+)\n\s+ADD CONSTRAINT (\S+) PRIMARY KEY \(([^)]+)\);", re.M)
_RE_FK = re.compile(
    r"^ALTER TABLE ONLY (\S+)\n\s+ADD CONSTRAINT (\S+) FOREIGN KEY \(([^)]+)\) REFERENCES (\S+)\(([^)]+)\)",
    re.M)
_RE_COMMENT = re.compile(r"^COMMENT ON (TABLE|COLUMN) (\S+) IS '((?:[^']|'')*)';", re.M | re.S)
_RE_VIEW = re.compile(r"^CREATE (?:OR REPLACE )?VIEW (\S+) AS", re.M)


def _strip_schema(qualified: str, pg_schema: str) -> str:
    """Elimina el prefijo ``pg_schema.`` y las comillas de un identificador."""
    name = qualified.strip().strip('"')
    prefix = pg_schema + "."
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name.strip('"')


def parse_pg_dump_schema(dump_path: Path, pg_schema: str, db_id: str) -> SchemaDict:
    """Extrae el esquema real de Catastro desde el ``pg_dump`` en texto plano.

    Lee el archivo por bloques (sin cargar los datos ``COPY`` en estructuras),
    parsea cada ``CREATE TABLE`` con SQLGlot (dialecto postgres) y recupera
    PK/FK de las sentencias ``ALTER TABLE ... ADD CONSTRAINT``.
    """
    if not dump_path.exists():
        raise FileNotFoundError(f"No se encontró el dump de Catastro: {dump_path}")

    # El dump es grande (26 MB) pero mayoritariamente datos COPY; se filtran
    # las líneas de datos para trabajar solo con el DDL.
    ddl_lines: list[str] = []
    in_copy = False
    with dump_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if in_copy:
                if line.startswith("\\."):
                    in_copy = False
                continue
            if line.startswith("COPY "):
                in_copy = True
                continue
            ddl_lines.append(line)
    ddl_text = "".join(ddl_lines)

    issues: list[str] = []
    tables: list[TableDict] = []
    columns: list[ColumnDict] = []
    ddl_statements: list[str] = []
    table_index_by_name: dict[str, int] = {}

    for stmt, qualified in _RE_CREATE_TABLE.findall(ddl_text):
        tname = _strip_schema(qualified, pg_schema)
        try:
            parsed = sqlglot.parse_one(stmt, read="postgres")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"No se pudo parsear CREATE TABLE {tname}: {exc}")
            continue
        t_idx = len(tables)
        table_index_by_name[tname.lower()] = t_idx
        tables.append({"table_index": t_idx, "name_original": tname, "name_normalized": tname.replace("_", " "),
                       "element_id": tname, "description": None, "n_columns": 0})
        for coldef in parsed.this.expressions:
            if not isinstance(coldef, exp.ColumnDef):
                continue  # CONSTRAINT CHECK, etc.
            cname = coldef.this.name
            ctype = coldef.args.get("kind")
            columns.append({
                "column_index": len(columns) + 1,
                "table_index": t_idx,
                "table_name": tname,
                "name_original": cname,
                "name_normalized": cname.replace("_", " "),
                "column_type": ctype.sql(dialect="postgres").lower() if ctype is not None else None,
                "element_id": f"{tname}.{cname}",
                "is_primary_key": False,
                "description": None,
            })
            tables[t_idx]["n_columns"] += 1
        ddl_statements.append(stmt + ";")

    col_lookup = {(c["table_name"].lower(), c["name_original"].lower()): c for c in columns}

    primary_keys: list[dict] = []
    for qualified, _cname, cols in _RE_PK.findall(ddl_text):
        tname = _strip_schema(qualified, pg_schema)
        col_names = [c.strip().strip('"') for c in cols.split(",")]
        for cn in col_names:
            c = col_lookup.get((tname.lower(), cn.lower()))
            if c is None:
                issues.append(f"PK sobre columna desconocida {tname}.{cn}")
            else:
                c["is_primary_key"] = True
        primary_keys.append({"table": tname, "columns": col_names})

    foreign_keys: list[dict] = []
    for qualified, _cname, from_cols, ref_table, to_cols in _RE_FK.findall(ddl_text):
        ft = _strip_schema(qualified, pg_schema)
        tt = _strip_schema(ref_table, pg_schema)
        f_cols = [c.strip().strip('"') for c in from_cols.split(",")]
        t_cols = [c.strip().strip('"') for c in to_cols.split(",")]
        for fc, tc in zip(f_cols, t_cols):
            if (ft.lower(), fc.lower()) not in col_lookup or (tt.lower(), tc.lower()) not in col_lookup:
                issues.append(f"FK con referencia no resoluble: {ft}.{fc} -> {tt}.{tc}")
                continue
            foreign_keys.append({"from_table": ft, "from_column": fc, "to_table": tt, "to_column": tc})

    # Comentarios (descripciones en español) de tablas y columnas
    table_by_lower = {t["name_original"].lower(): t for t in tables}
    n_comments = 0
    for kind, target, text in _RE_COMMENT.findall(ddl_text):
        text = text.replace("''", "'").strip()
        name = _strip_schema(target, pg_schema)
        if kind == "TABLE":
            t = table_by_lower.get(name.lower())
            if t is not None:
                t["description"] = text
                n_comments += 1
        else:
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                c = col_lookup.get((parts[0].lower(), parts[1].lower()))
                if c is not None:
                    c["description"] = text
                    n_comments += 1

    views = [_strip_schema(v, pg_schema) for v in _RE_VIEW.findall(ddl_text)]

    schema: SchemaDict = {
        "schema_id": f"CATASTRO:{db_id}",
        "db_id": db_id,
        "source": "CATASTRO",
        "dialect": config.DIALECTS["CATASTRO"],
        "pg_schema": pg_schema,
        "tables": tables,
        "columns": columns,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "relations": _build_relations(foreign_keys),
        "ddl": "\n".join(ddl_statements),
        "ddl_origin": "pg_dump",
        "views": views,
        "n_tables": len(tables),
        "n_columns": len(columns),
        "issues": issues,
    }
    logger.info("Catastro: %d tablas, %d columnas, %d PK, %d FK, %d comentarios, %d vistas (no candidatas)",
                len(tables), len(columns), len(primary_keys), len(foreign_keys), n_comments, len(views))
    return schema


# ---------------------------------------------------------------------------
# Catálogo completo
# ---------------------------------------------------------------------------
def build_schema_catalog(save: bool = True, required_schema_ids: set[str] | None = None) -> dict[str, SchemaDict]:
    """Construye ``schemas.json`` con todos los esquemas de las tres fuentes.

    Args:
        save: persiste el catálogo en ``data/processed/schemas.json``.
        required_schema_ids: si se indica, se registran como incidencias los
            esquemas requeridos por el dataset maestro que no se encuentren.

    Returns:
        Diccionario ``schema_id -> esquema``.
    """
    config.ensure_directories()
    catalog: dict[str, SchemaDict] = {}

    for schema in iter_benchmark_schemas(config.SPIDER_FILES["tables"], "SPIDER"):
        catalog[schema["schema_id"]] = schema
    logger.info("Spider: %d esquemas", sum(1 for s in catalog.values() if s["source"] == "SPIDER"))

    for key in ("train_tables", "dev_tables"):
        for schema in iter_benchmark_schemas(config.BIRD_FILES[key], "BIRD"):
            if schema["schema_id"] in catalog:
                catalog[schema["schema_id"]]["issues"].append(f"db_id repetido en {key}")
            catalog[schema["schema_id"]] = schema
    logger.info("BIRD: %d esquemas", sum(1 for s in catalog.values() if s["source"] == "BIRD"))

    try:
        cat = parse_pg_dump_schema(config.CATASTRO_FILES["dump_sql"], config.CATASTRO_PG_SCHEMA,
                                   config.CATASTRO_DB_ID)
        catalog[cat["schema_id"]] = cat
    except FileNotFoundError as exc:
        logger.error("Esquema de Catastro no disponible: %s", exc)

    if required_schema_ids:
        missing = sorted(required_schema_ids - set(catalog))
        if missing:
            logger.error("Esquemas requeridos no encontrados: %s", missing)

    n_issues = sum(len(s["issues"]) for s in catalog.values())
    logger.info("Catálogo: %d esquemas, %d incidencias registradas", len(catalog), n_issues)

    if save:
        with config.SCHEMAS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(catalog, fh, ensure_ascii=False, indent=1)
        logger.info("Guardado %s", relative_to_project(config.SCHEMAS_PATH))
    return catalog


def load_schema_catalog(path: Path = config.SCHEMAS_PATH) -> dict[str, SchemaDict]:
    """Carga el catálogo de esquemas desde ``schemas.json``."""
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute build_schema_catalog() primero.")
    return load_json(path)


def schema_summary(catalog: dict[str, SchemaDict]):
    """DataFrame resumen (una fila por esquema) para analítica descriptiva."""
    import pandas as pd

    return pd.DataFrame([
        {"schema_id": s["schema_id"], "source": s["source"], "db_id": s["db_id"],
         "n_tables": s["n_tables"], "n_columns": s["n_columns"],
         "n_primary_keys": len(s["primary_keys"]), "n_foreign_keys": len(s["foreign_keys"]),
         "n_issues": len(s["issues"])}
        for s in catalog.values()
    ])


if __name__ == "__main__":
    build_schema_catalog(save=True)

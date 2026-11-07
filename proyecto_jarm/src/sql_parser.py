"""Análisis sintáctico (AST con SQLGlot) y resolución semántica de SQL.

Para cada consulta se construye el AST, se recorren sus *scopes* (bloques de
consulta: consulta principal, subconsultas, CTE, ramas de UNION) y se resuelve
cada identificador contra el esquema real:

* Tablas: se resuelven por nombre (insensible a mayúsculas) y se les asignan
  los roles ``FROM`` / ``JOIN`` según el bloque en el que aparecen.
* Columnas: se resuelven considerando alias, calificación, bloque de consulta,
  tablas derivadas (subconsultas y CTE, mediante linaje de proyecciones) y
  scopes externos (subconsultas correlacionadas). Si una columna no calificada
  existe en varias tablas del bloque y no puede desambiguarse, se registra como
  ``unresolved`` (nunca se asigna la primera tabla disponible).
* Cláusula: se determina caminando por los ancestros del nodo hasta el bloque
  ``SELECT`` que lo contiene; los roles son locales a cada bloque.

Convenciones metodológicas (documentadas también en el notebook 03):
    - El comodín ``*`` no genera columnas; se registra como ``star`` por tabla.
    - En SQLite un identificador entre comillas dobles que no corresponde a
      ninguna columna se interpreta como literal de texto (comportamiento real
      de SQLite); se registra como ``literal_fallback`` y no como error.
    - Cuando una columna de la consulta externa referencia la salida de un CTE
      o subconsulta, el rol de la cláusula externa se propaga a las columnas
      físicas de origen (``via_derived``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from src import config
from src.config import get_logger
from src.data_loader import relative_to_project

logger = get_logger(__name__, "sql_parser.log")

# Cláusulas que producen roles (en el orden canónico de ROLE_LABELS)
ROLE_SET = set(config.ROLE_LABELS)
_ARG_TO_CLAUSE = {
    "expressions": "SELECT",
    "distinct": "SELECT",
    "windows": "SELECT",
    "qualify": "SELECT",
    "where": "WHERE",
    "group": "GROUP_BY",
    "having": "HAVING",
    "order": "ORDER_BY",
    "limit": "LIMIT",
    "offset": "LIMIT",
    "from_": "FROM",
    "joins": "JOIN",
}

AGG_NAMES = {"COUNT", "SUM", "AVG", "MIN", "MAX", "TOTAL", "GROUP_CONCAT", "STRING_AGG",
             "ARRAY_AGG", "STDDEV", "VARIANCE", "MEDIAN", "COUNT_IF"}


# ---------------------------------------------------------------------------
# Índice de esquema para resolución rápida
# ---------------------------------------------------------------------------
@dataclass
class SchemaIndex:
    """Índice en minúsculas de un esquema del catálogo para resolver identificadores."""

    schema_id: str
    pg_schema: str | None
    tables: dict[str, str]                        # lower -> nombre original
    columns: dict[str, dict[str, str]]            # table_lower -> {col_lower: col_original}
    tables_with_column: dict[str, list[str]]      # col_lower -> [tablas originales]

    @classmethod
    def from_schema(cls, schema: dict) -> "SchemaIndex":
        tables = {t["name_original"].lower(): t["name_original"] for t in schema["tables"]}
        columns: dict[str, dict[str, str]] = {t: {} for t in tables}
        twc: dict[str, list[str]] = {}
        for c in schema["columns"]:
            tl, cl = c["table_name"].lower(), c["name_original"].lower()
            columns.setdefault(tl, {})[cl] = c["name_original"]
            twc.setdefault(cl, []).append(c["table_name"])
        return cls(schema["schema_id"], schema.get("pg_schema"), tables, columns, twc)

    def table(self, name: str) -> str | None:
        return self.tables.get(name.lower())

    def column(self, table: str, column: str) -> str | None:
        """Devuelve ``tabla.columna`` original si existe, si no ``None``."""
        t = self.table(table)
        if t is None:
            return None
        c = self.columns.get(t.lower(), {}).get(column.lower())
        return f"{t}.{c}" if c is not None else None


# ---------------------------------------------------------------------------
# Estructuras internas de resolución
# ---------------------------------------------------------------------------
@dataclass
class DerivedOutput:
    """Salida (proyecciones) de un CTE / subconsulta usada para linaje de columnas."""

    columns: dict[str, list[str]] = field(default_factory=dict)  # alias_lower -> element_ids físicos
    star_tables: list[str] = field(default_factory=list)          # tablas expuestas por '*'


@dataclass
class ScopeContext:
    """Fuentes visibles en un bloque de consulta."""

    scope: Scope
    # Fuentes seleccionadas (FROM/JOIN): candidatas para columnas no calificadas
    physical: dict[str, str] = field(default_factory=dict)         # alias_lower -> tabla original
    derived: dict[str, DerivedOutput] = field(default_factory=dict)  # alias_lower -> salida derivada
    derived_pending: dict[str, Scope] = field(default_factory=dict)  # alias_lower -> scope aún no procesado
    # Todas las fuentes visibles (incluye CTE declarados): para referencias calificadas
    physical_all: dict[str, str] = field(default_factory=dict)
    derived_pending_all: dict[str, Scope] = field(default_factory=dict)
    projection_aliases: dict[str, exp.Expression] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class ColumnResolution:
    raw: str
    clause: str
    element_ids: list[str]
    method: str
    via_derived: bool = False
    note: str | None = None


# ---------------------------------------------------------------------------
# Utilidades sobre el AST
# ---------------------------------------------------------------------------
def clause_of(node: exp.Expression) -> str:
    """Determina la cláusula (local al bloque) a la que pertenece un nodo."""
    child, parent = node, node.parent
    while parent is not None:
        if isinstance(parent, exp.Join):
            if child.arg_key in ("on", "using"):
                return "JOIN"
        if isinstance(parent, (exp.Select, exp.SetOperation)):
            return _ARG_TO_CLAUSE.get(child.arg_key or "", f"OTHER:{child.arg_key}")
        child, parent = parent, parent.parent
    return "OTHER:unknown"


def _is_quoted(col: exp.Column) -> bool:
    ident = col.this
    return isinstance(ident, exp.Identifier) and bool(ident.quoted)


def _aggregations(tree: exp.Expression) -> list[str]:
    names: list[str] = []
    for node in tree.find_all(exp.AggFunc):
        names.append(type(node).__name__.upper())
    for node in tree.find_all(exp.Anonymous):
        if str(node.name).upper() in AGG_NAMES:
            names.append(str(node.name).upper())
    return sorted(set(names))


def _set_operations(tree: exp.Expression) -> list[str]:
    ops = []
    for node in tree.find_all(exp.SetOperation):
        name = type(node).__name__.upper()
        if node.args.get("distinct") is False:
            name += " ALL"
        ops.append(name)
    return ops


# ---------------------------------------------------------------------------
# Analizador
# ---------------------------------------------------------------------------
class SQLAnalyzer:
    """Analiza una consulta SQL frente a un esquema y extrae tablas, columnas y roles."""

    def __init__(self, schema_index: SchemaIndex, dialect: str):
        self.idx = schema_index
        self.dialect = dialect
        self.table_roles: dict[str, set[str]] = {}
        self.column_roles: dict[str, set[str]] = {}
        self.column_refs: list[dict[str, Any]] = []
        self.table_refs: list[dict[str, Any]] = []
        self.joins: list[dict[str, Any]] = []
        self.unresolved: list[dict[str, Any]] = []
        self.literal_fallbacks: list[str] = []
        self.stars: list[dict[str, Any]] = []
        self.ctes: list[str] = []
        self.aliases: dict[str, str] = {}
        self._ctx: dict[int, ScopeContext] = {}
        self._resolved_nodes: dict[int, list[str]] = {}
        self._seen: set[int] = set()
        self._outputs: dict[int, DerivedOutput] = {}
        self.n_scopes = 0
        self.n_subqueries = 0

    # -- helpers ------------------------------------------------------------
    def _add_table_role(self, table: str, roles: Iterable[str]) -> None:
        self.table_roles.setdefault(table, set()).update(roles)

    def _add_column_roles(self, element_ids: Iterable[str], clause: str) -> None:
        for eid in element_ids:
            roles = self.column_roles.setdefault(eid, set())
            if clause in ROLE_SET:
                roles.add(clause)
            table = eid.split(".", 1)[0]
            self.table_roles.setdefault(table, set())

    def _unresolved(self, kind: str, raw: str, reason: str, clause: str | None = None,
                    candidates: list[str] | None = None) -> None:
        self.unresolved.append({"kind": kind, "raw": raw, "reason": reason, "clause": clause,
                                "candidates": candidates or []})

    def _resolve_table_node(self, table: exp.Table) -> str | None:
        """Resuelve un nodo ``Table`` físico contra el esquema."""
        name = table.name
        if table.db and self.idx.pg_schema and table.db.lower() != self.idx.pg_schema.lower():
            self._unresolved("table", table.sql(), f"schema '{table.db}' distinto de '{self.idx.pg_schema}'")
            return None
        resolved = self.idx.table(name)
        if resolved is None:
            self._unresolved("table", table.sql(), "table_not_in_schema")
        return resolved

    # -- paso 1: fuentes por scope ------------------------------------------
    def _build_contexts(self, scopes: list[Scope]) -> None:
        for scope in scopes:
            ctx = ScopeContext(scope=scope)
            try:
                selected = {k: v for k, (_node, v) in scope.selected_sources.items()}
            except Exception as exc:  # noqa: BLE001  (p.ej. alias duplicado en el SQL gold)
                selected = dict(scope.sources)
                ctx.notes.append(f"selected_sources_error: {str(exc)[:120]}")
                self._unresolved("scope", scope.expression.sql(dialect=self.dialect)[:80],
                                 f"duplicate_alias: {str(exc)[:120]}")
            resolved_cache: dict[int, str | None] = {}
            for alias, source in scope.sources.items():
                key = (alias or "").lower()
                is_selected = alias in selected
                if isinstance(source, exp.Table):
                    if id(source) not in resolved_cache:
                        resolved_cache[id(source)] = self._resolve_table_node(source)
                    resolved = resolved_cache[id(source)]
                    if resolved is not None:
                        ctx.physical_all[key] = resolved
                        if is_selected:
                            ctx.physical[key] = resolved
                        if source.alias:
                            self.aliases[source.alias] = resolved
                elif isinstance(source, Scope):
                    ctx.derived_pending_all[key] = source
                    if is_selected:
                        ctx.derived_pending[key] = source
            selects_for_alias: list[exp.Expression] = []
            if isinstance(scope.expression, exp.Select):
                selects_for_alias = scope.expression.selects
            elif isinstance(scope.expression, exp.SetOperation) and scope.set_operation_scopes:
                left = scope.set_operation_scopes[0].expression
                selects_for_alias = left.selects if isinstance(left, exp.Select) else []
            for proj in selects_for_alias:
                if isinstance(proj, exp.Alias):
                    ctx.projection_aliases[proj.alias.lower()] = proj
            self._ctx[id(scope)] = ctx

    def _ctx_of(self, scope: Scope) -> ScopeContext:
        return self._ctx[id(scope)]

    def _derived_output(self, ctx: ScopeContext, key: str, include_all: bool = False) -> DerivedOutput | None:
        """Salida de una fuente derivada (CTE/subconsulta) si su scope ya fue procesado."""
        if key in ctx.derived:
            return ctx.derived[key]
        pending = ctx.derived_pending.get(key)
        if pending is None and include_all:
            pending = ctx.derived_pending_all.get(key)
        if pending is not None and id(pending) in self._outputs:
            out = self._outputs[id(pending)]
            if key in ctx.derived_pending:
                ctx.derived[key] = out
            return out
        return None

    def _all_derived(self, ctx: ScopeContext) -> dict[str, DerivedOutput]:
        """Salidas de todas las fuentes derivadas *seleccionadas* en el bloque."""
        for key in list(ctx.derived_pending):
            self._derived_output(ctx, key)
        return ctx.derived

    # -- paso 2: tablas y roles FROM / JOIN ------------------------------------
    def _process_tables(self, scope: Scope) -> None:
        expr = scope.expression
        if not isinstance(expr, exp.Select):
            return
        from_ = expr.args.get("from_")
        joins = expr.args.get("joins") or []
        has_join = len(joins) > 0
        ctx = self._ctx_of(scope)

        def register(node: exp.Expression, roles: set[str], position: str, join: exp.Join | None) -> None:
            if isinstance(node, exp.Table):
                alias_key = (node.alias_or_name or "").lower()
                if alias_key in ctx.derived_pending_all or alias_key in ctx.derived:
                    # Referencia a un CTE: los roles se asignan dentro del propio CTE
                    self.table_refs.append({"raw": node.sql(), "table": None, "alias": node.alias or None,
                                            "position": position, "kind": "cte_reference"})
                    return
                resolved = ctx.physical.get(alias_key) or self.idx.table(node.name)
                if resolved is None:
                    return  # ya registrado como unresolved en _build_contexts
                self._add_table_role(resolved, roles)
                self.table_refs.append({"raw": node.sql(), "table": resolved, "alias": node.alias or None,
                                        "position": position, "kind": "table"})
            elif isinstance(node, exp.Subquery):
                self.table_refs.append({"raw": "(subquery)", "table": None, "alias": node.alias or None,
                                        "position": position, "kind": "derived_table"})

        if from_ is not None:
            roles = {"FROM", "JOIN"} if has_join else {"FROM"}
            register(from_.this, roles, "FROM", None)
        for join in joins:
            register(join.this, {"FROM", "JOIN"}, "JOIN", join)
            self.joins.append({
                "target": join.this.alias_or_name if hasattr(join.this, "alias_or_name") else join.this.sql(),
                "kind": (join.side + " " + join.kind).strip() or ("CROSS/IMPLICIT" if not join.args.get("on")
                                                                     and not join.args.get("using") else "INNER"),
                "on": join.args["on"].sql(dialect=self.dialect) if join.args.get("on") else None,
                "using": [i.name for i in join.args.get("using") or []],
                "on_columns": [],
            })
            # USING (col): la columna existe en ambas tablas unidas
            for ident in join.args.get("using") or []:
                self._resolve_using_column(ctx, ident.name, join)

    def _resolve_using_column(self, ctx: ScopeContext, name: str, join: exp.Join) -> None:
        found = []
        for alias_key, table in ctx.physical.items():
            eid = self.idx.column(table, name)
            if eid:
                found.append(eid)
        if found:
            self._add_column_roles(found, "JOIN")
            self.column_refs.append({"raw": f"USING({name})", "clause": "JOIN", "element_ids": found,
                                     "method": "using", "via_derived": False})
            self.joins[-1]["on_columns"].extend(found)
        else:
            self._unresolved("column", f"USING({name})", "column_not_found", "JOIN")

    # -- paso 3: columnas -----------------------------------------------------
    def _lookup_alias(self, scope: Scope, alias_key: str) -> tuple[str, Any] | None:
        """Busca un alias/tabla en el scope actual y en los externos (correlación)."""
        current: Scope | None = scope
        hops = 0
        while current is not None:
            ctx = self._ctx.get(id(current))
            if ctx is not None:
                if alias_key in ctx.physical_all:
                    return ("physical" if hops == 0 else "physical_outer", ctx.physical_all[alias_key])
                out = self._derived_output(ctx, alias_key, include_all=True)
                if out is not None:
                    return ("derived" if hops == 0 else "derived_outer", out)
                if alias_key in ctx.derived_pending_all:
                    return ("derived_pending", None)
                # Columna calificada con el nombre real de una tabla que tiene alias
                for table in set(ctx.physical_all.values()):
                    if table.lower() == alias_key:
                        return ("physical_by_table_name" if hops == 0 else "physical_outer", table)
            current = current.parent
            hops += 1
        return None

    def _lineage_from_projection(self, proj: exp.Expression) -> list[str]:
        ids: list[str] = []
        for col in proj.find_all(exp.Column):
            ids.extend(self._resolved_nodes.get(id(col), []))
        return sorted(set(ids))

    def _resolve_unqualified(self, scope: Scope, name: str, clause: str) -> tuple[list[str], str, bool] | None:
        """Resuelve una columna no calificada. Devuelve (element_ids, método, via_derived) o None."""
        name_l = name.lower()
        current: Scope | None = scope
        hops = 0
        while current is not None:
            ctx = self._ctx.get(id(current))
            if ctx is None:
                current = current.parent
                hops += 1
                continue
            suffix = "" if hops == 0 else "_outer"
            alias_proj = ctx.projection_aliases.get(name_l)
            # En ORDER BY (y HAVING/GROUP BY) los alias de proyección tienen prioridad
            if alias_proj is not None and clause in ("ORDER_BY", "HAVING", "GROUP_BY"):
                return self._lineage_from_projection(alias_proj), "projection_alias" + suffix, True

            phys = sorted({self.idx.column(t, name) for t in ctx.physical.values() if self.idx.column(t, name)})
            derived_hits: list[tuple[str, list[str]]] = []
            for key, out in self._all_derived(ctx).items():
                if name_l in out.columns:
                    derived_hits.append((key, out.columns[name_l]))
                else:
                    for t in out.star_tables:
                        eid = self.idx.column(t, name)
                        if eid:
                            derived_hits.append((key, [eid]))
            total = len(phys) + len(derived_hits)
            if total == 1:
                if phys:
                    return phys, "unqualified_unique" + suffix, False
                return derived_hits[0][1], "unqualified_derived" + suffix, True
            if total > 1:
                cands = phys + [f"{k}.{name}" for k, _ in derived_hits]
                self._unresolved("column", name, "ambiguous", clause, cands)
                return [], "ambiguous", False
            if alias_proj is not None:
                return self._lineage_from_projection(alias_proj), "projection_alias" + suffix, True
            current = current.parent
            hops += 1
        return None

    def _resolve_column(self, scope: Scope, col: exp.Column, clause: str) -> ColumnResolution | None:
        raw = col.sql(dialect=self.dialect)
        name = col.name
        if col.table:
            hit = self._lookup_alias(scope, col.table.lower())
            if hit is None:
                # Tabla calificada con esquema (schema.table.col) sin alias en el scope
                if col.db and self.idx.table(col.table):
                    eid = self.idx.column(col.table, name)
                    if eid:
                        return ColumnResolution(raw, clause, [eid], "qualified_schema_table")
                self._unresolved("column", raw, "alias_not_found", clause)
                return None
            kind, target = hit
            if kind == "derived_pending":
                self._unresolved("column", raw, "derived_source_not_available", clause)
                return None
            if kind.startswith("physical"):
                eid = self.idx.column(target, name)
                if eid is None:
                    self._unresolved("column", raw, f"column_not_in_table:{target}", clause)
                    return None
                return ColumnResolution(raw, clause, [eid], "qualified_" + kind)
            out: DerivedOutput = target
            if name.lower() in out.columns:
                return ColumnResolution(raw, clause, out.columns[name.lower()], "qualified_" + kind, True)
            for t in out.star_tables:
                eid = self.idx.column(t, name)
                if eid:
                    return ColumnResolution(raw, clause, [eid], "qualified_derived_star", True)
            self._unresolved("column", raw, "column_not_in_derived_source", clause)
            return None

        res = self._resolve_unqualified(scope, name, clause)
        if res is None:
            if _is_quoted(col) and self.dialect == "sqlite":
                self.literal_fallbacks.append(raw)
                return ColumnResolution(raw, clause, [], "literal_fallback", note="sqlite_double_quoted_literal")
            self._unresolved("column", raw, "column_not_found", clause)
            return None
        ids, method, via = res
        if method == "ambiguous":
            return None
        return ColumnResolution(raw, clause, ids, method, via)

    def _process_columns(self, scope: Scope) -> None:
        for col in scope.columns:
            if id(col) in self._seen:
                continue
            self._seen.add(id(col))
            clause = clause_of(col)
            resolution = self._resolve_column(scope, col, clause)
            if resolution is None:
                continue
            self._resolved_nodes[id(col)] = resolution.element_ids
            if resolution.element_ids:
                self._add_column_roles(resolution.element_ids, clause)
            self.column_refs.append({
                "raw": resolution.raw, "clause": clause, "element_ids": resolution.element_ids,
                "method": resolution.method, "via_derived": resolution.via_derived,
            })
            if clause == "JOIN" and self.joins:
                # Asociar al JOIN más cercano por posición textual
                join = col.find_ancestor(exp.Join)
                if join is not None:
                    for j in self.joins:
                        if j["on"] and join.args.get("on") is not None and \
                                j["on"] == join.args["on"].sql(dialect=self.dialect):
                            j["on_columns"].extend(resolution.element_ids)
                            break

    def _process_stars(self, scope: Scope) -> None:
        expr = scope.expression
        if not isinstance(expr, exp.Select):
            return
        ctx = self._ctx_of(scope)
        for proj in expr.selects:
            if isinstance(proj, exp.Star):
                self.stars.append({"raw": "*", "clause": "SELECT", "tables": sorted(set(ctx.physical.values()))})
            elif isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star):
                hit = self._lookup_alias(scope, proj.table.lower()) if proj.table else None
                if hit and hit[0].startswith("physical"):
                    self.stars.append({"raw": proj.sql(), "clause": "SELECT", "tables": [hit[1]]})
                else:
                    self.stars.append({"raw": proj.sql(), "clause": "SELECT", "tables": []})
        seen_local: set[tuple[str, str]] = set()
        for node in expr.find_all(exp.Star):
            # COUNT(*) y similares: participación de las tablas del bloque sin columnas concretas.
            # Solo se consideran los '*' cuyo bloque SELECT más cercano es el actual.
            if node.parent is expr or isinstance(node.parent, exp.Column):
                continue
            if node.find_ancestor(exp.Select) is not expr:
                continue
            key = (node.parent.sql(dialect=self.dialect)[:60], clause_of(node))
            if key in seen_local:
                continue
            seen_local.add(key)
            self.stars.append({"raw": key[0], "clause": key[1], "tables": sorted(set(ctx.physical.values()))})

    def _build_output(self, scope: Scope) -> DerivedOutput:
        out = DerivedOutput()
        expr = scope.expression
        if isinstance(expr, exp.SetOperation):
            branches = scope.set_operation_scopes or []
            if branches and id(branches[0]) in self._outputs:
                return self._outputs[id(branches[0])]
            return out
        if not isinstance(expr, exp.Select):
            return out
        ctx = self._ctx_of(scope)
        for proj in expr.selects:
            if isinstance(proj, exp.Star):
                out.star_tables.extend(sorted(set(ctx.physical.values())))
                for d in self._all_derived(ctx).values():
                    out.columns.update(d.columns)
                    out.star_tables.extend(d.star_tables)
            elif isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star):
                hit = self._lookup_alias(scope, proj.table.lower()) if proj.table else None
                if hit and hit[0].startswith("physical"):
                    out.star_tables.append(hit[1])
                elif hit and hit[0].startswith("derived"):
                    out.columns.update(hit[1].columns)
                    out.star_tables.extend(hit[1].star_tables)
            else:
                out.columns[proj.alias_or_name.lower()] = self._lineage_from_projection(proj)
        return out

    # -- orquestación ---------------------------------------------------------
    def analyze(self, tree: exp.Expression) -> dict[str, Any]:
        scopes = list(traverse_scope(tree))
        self.n_scopes = len(scopes)
        self._build_contexts(scopes)
        for scope in scopes:
            if scope.is_cte:
                name = scope.expression.parent.alias if isinstance(scope.expression.parent, exp.CTE) else None
                if name:
                    self.ctes.append(name)
            elif not scope.is_root and not isinstance(scope.expression, exp.SetOperation):
                if scope.parent is None or not isinstance(scope.parent.expression, exp.SetOperation):
                    self.n_subqueries += 1
            self._process_tables(scope)
            self._process_stars(scope)
            self._process_columns(scope)
            self._outputs[id(scope)] = self._build_output(scope)

        # Tablas referenciadas solo por '*' siguen siendo relevantes (ya tienen rol FROM)
        for eid in self.column_roles:
            self.table_roles.setdefault(eid.split(".", 1)[0], set())

        return {
            "table_roles": {t: sorted(r) for t, r in sorted(self.table_roles.items())},
            "column_roles": {c: sorted(r) for c, r in sorted(self.column_roles.items())},
            "table_refs": self.table_refs,
            "column_refs": self.column_refs,
            "aliases": self.aliases,
            "joins": self.joins,
            "ctes": self.ctes,
            "stars": self.stars,
            "unresolved": self.unresolved,
            "literal_fallbacks": self.literal_fallbacks,
            "n_scopes": self.n_scopes,
            "n_subqueries": self.n_subqueries,
        }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def parse_sql(sql: str, dialect: str) -> tuple[exp.Expression | None, str | None]:
    """Parsea una consulta con SQLGlot; devuelve (AST, error)."""
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:300]}"
    statements = [s for s in statements if s is not None]
    if not statements:
        return None, "empty_statement"
    if len(statements) > 1:
        return statements[0], f"multiple_statements:{len(statements)}"
    return statements[0], None


def analyze_sql(sql: str, dialect: str, schema_index: SchemaIndex) -> dict[str, Any]:
    """Analiza una consulta y devuelve el diccionario completo de análisis."""
    tree, error = parse_sql(sql, dialect)
    result: dict[str, Any] = {
        "parse_status": "ok" if tree is not None else "parse_error",
        "parse_error": error,
        "resolution_status": "parse_error",
    }
    if tree is None:
        return result
    try:
        analyzer = SQLAnalyzer(schema_index, dialect)
        details = analyzer.analyze(tree)
    except Exception as exc:  # noqa: BLE001
        result.update({"parse_status": "analysis_error", "parse_error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return result

    n_unres_cols = sum(1 for u in details["unresolved"] if u["kind"] == "column")
    n_unres_tabs = sum(1 for u in details["unresolved"] if u["kind"] == "table")
    n_resolved = len(details["column_roles"]) + len(details["table_roles"])
    if not details["unresolved"]:
        status = "resolved"
    elif n_resolved > 0:
        status = "partial"
    else:
        status = "unresolved"

    result.update({
        "resolution_status": status,
        "n_tables_used": len(details["table_roles"]),
        "n_columns_used": len(details["column_roles"]),
        "n_unresolved_columns": n_unres_cols,
        "n_unresolved_tables": n_unres_tabs,
        "n_literal_fallbacks": len(details["literal_fallbacks"]),
        "n_subqueries": details["n_subqueries"],
        "n_ctes": len(details["ctes"]),
        "n_joins": len(details["joins"]),
        "has_join": len(details["joins"]) > 0,
        "has_subquery": details["n_subqueries"] > 0,
        "has_cte": len(details["ctes"]) > 0,
        "has_where": tree.find(exp.Where) is not None,
        "has_group_by": tree.find(exp.Group) is not None,
        "has_having": tree.find(exp.Having) is not None,
        "has_order_by": tree.find(exp.Order) is not None,
        "has_limit": tree.find(exp.Limit) is not None,
        "has_distinct": tree.find(exp.Distinct) is not None,
        "has_star": len(details["stars"]) > 0,
        "aggregations": _aggregations(tree),
        "has_aggregation": len(_aggregations(tree)) > 0,
        "set_operations": _set_operations(tree),
        "has_set_operation": tree.find(exp.SetOperation) is not None,
        "has_window": tree.find(exp.Window) is not None,
        "details": details,
    })
    return result


_FLAT_COLUMNS = [
    "parse_status", "parse_error", "resolution_status", "n_tables_used", "n_columns_used",
    "n_unresolved_columns", "n_unresolved_tables", "n_literal_fallbacks", "n_subqueries", "n_ctes",
    "n_joins", "has_join", "has_subquery", "has_cte", "has_where", "has_group_by", "has_having",
    "has_order_by", "has_limit", "has_distinct", "has_star", "has_aggregation", "has_set_operation",
    "has_window",
]


def analyze_dataset(master: pd.DataFrame, catalog: dict[str, dict], save: bool = True,
                    sources: Iterable[str] | None = None) -> pd.DataFrame:
    """Analiza todas las consultas del dataset maestro y genera ``sql_analysis.parquet``.

    Los índices de esquema se construyen bajo demanda (uno por ``schema_id``)
    para no cargar estructuras innecesarias en memoria.
    """
    config.ensure_directories()
    sources = tuple(sources) if sources else config.SOURCES
    rows: list[dict[str, Any]] = []
    index_cache: dict[str, SchemaIndex] = {}
    subset = master[master.source.isin(sources)]
    total = len(subset)
    for i, rec in enumerate(subset.itertuples(index=False), start=1):
        sid = rec.schema_id
        if sid not in index_cache:
            if sid not in catalog:
                rows.append({"example_id": rec.example_id, "source": rec.source, "db_id": rec.db_id,
                             "dialect": rec.dialect, "parse_status": "no_schema", "parse_error": "schema_not_found",
                             "resolution_status": "no_schema"})
                continue
            index_cache[sid] = SchemaIndex.from_schema(catalog[sid])
        res = analyze_sql(rec.sql_gold, rec.dialect, index_cache[sid])
        row = {"example_id": rec.example_id, "source": rec.source, "db_id": rec.db_id, "dialect": rec.dialect}
        for c in _FLAT_COLUMNS:
            row[c] = res.get(c)
        row["aggregations"] = json.dumps(res.get("aggregations", []))
        row["set_operations"] = json.dumps(res.get("set_operations", []))
        row["analysis_json"] = json.dumps(res.get("details", {}), ensure_ascii=False)
        rows.append(row)
        if i % 2000 == 0 or i == total:
            logger.info("Analizadas %d/%d consultas", i, total)

    df = pd.DataFrame(rows)
    logger.info("Estado de parseo:\n%s", df.groupby(["source", "parse_status"]).size().to_string())
    logger.info("Estado de resolución:\n%s", df.groupby(["source", "resolution_status"]).size().to_string())
    if save:
        df.to_parquet(config.SQL_ANALYSIS_PATH, index=False)
        logger.info("Guardado %s", relative_to_project(config.SQL_ANALYSIS_PATH))
    return df


def load_sql_analysis(path: Path = config.SQL_ANALYSIS_PATH) -> pd.DataFrame:
    """Carga ``sql_analysis.parquet``."""
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute analyze_dataset() primero.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    from src.data_loader import load_master_dataset
    from src.schema_parser import load_schema_catalog

    analyze_dataset(load_master_dataset(), load_schema_catalog(), save=True)

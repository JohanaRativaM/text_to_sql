"""Paquete de preparación de datos para Text-to-SQL con Schema Linking.

Módulos:
    config            : rutas, constantes y configuración centralizada.
    data_loader       : carga y consolidación de Spider, BIRD y Catastro.
    schema_parser     : catálogo homogéneo de esquemas (schemas.json).
    sql_parser        : análisis sintáctico y resolución semántica de SQL (SQLGlot).
    schema_annotator  : etiquetas de relevancia y roles para Schema Linking.
    dataset_builder   : corpus JSONL para clasificación (y plantilla del generador).
    dataset_validator : controles automáticos de calidad.
    statistics        : analítica descriptiva y gráficos.
"""

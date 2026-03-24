from typing import List
from app.models.schemas import AnalyzeRow

HEADER = (
    "ID_CONVERSACION;Tipo;SubTipo;CodTipo;CodSubTipo;Resolucion;"
    "Satisfaccion;Loop;Falla_IA;Compra_Tarjeta;Compra_AV_SAV;"
    "Compra_Seguro;Opcion_Pago"
)

def escape_csv_field(value: str) -> str:
    value = value.replace("\n", " ").replace("\r", " ").strip()
    if ";" in value or '"' in value:
        value = value.replace('"', '""')
        return f'"{value}"'
    return value

def rows_to_csv(rows: List[AnalyzeRow]) -> str:
    lines = [HEADER]
    for r in rows:
        lines.append(
            ";".join(
                [
                    escape_csv_field(r.id_conversacion),
                    escape_csv_field(r.tipo),
                    escape_csv_field(r.subtipo),
                    str(r.cod_tipo),
                    str(r.cod_subtipo),
                    r.resolucion,
                    r.satisfaccion,
                    str(r.loop),
                    r.falla_ia,
                    str(r.compra_tarjeta),
                    str(r.compra_av_sav),
                    str(r.compra_seguro),
                    str(r.opcion_pago),
                ]
            )
        )
    return "\n".join(lines)

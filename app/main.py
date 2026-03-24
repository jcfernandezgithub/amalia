from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.models.schemas import AnalyzeRequest, AnalyzeResponse, CallInput
from app.services.classifier import ClassifierService
from app.services.csv_exporter import rows_to_csv
from app.services.dictionary_service import DictionaryService

app = FastAPI(title="Amalia CX Auditor", version="1.0.0")

dictionary_service = DictionaryService()
classifier_service = ClassifierService(dictionary_service.all())

EXPECTED_COLUMNS = [
    "ID_CONVERSACION",
    "REQUEST_TIME",
    "RUT_CLIENTE",
    "PHONO_CONTACTO",
    "FIN_LLAMADA",
    "MARCA_ABANDONO",
    "MARCA_DERIVADO",
    "COD_TIPIFICACION",
    "IVR",
    "CONVERSACION",
]


@app.get("/health")
def health():
    return {"ok": True, "service": "amalia-auditor"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest):
    rows = [classifier_service.classify(call) for call in payload.calls]
    csv_content = rows_to_csv(rows)
    return AnalyzeResponse(total=len(rows), rows=rows, csv=csv_content)


@app.post("/analyze-file", response_class=PlainTextResponse)
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Debes subir un archivo CSV")

    content = await file.read()

    df = None
    used_sep = None
    read_errors = []

    for sep in [",", ";", "\t"]:
        try:
            temp_df = pd.read_csv(BytesIO(content), sep=sep, encoding="utf-8-sig")
            temp_df.columns = [str(col).replace("\ufeff", "").strip() for col in temp_df.columns]

            # caso normal: sí trae headers
            if "ID_CONVERSACION" in temp_df.columns and "CONVERSACION" in temp_df.columns:
                df = temp_df
                used_sep = sep
                break

            # fallback: leer sin header
            temp_df_no_header = pd.read_csv(
                BytesIO(content),
                sep=sep,
                encoding="utf-8-sig",
                header=None
            )

            if temp_df_no_header.shape[1] >= 10:
                temp_df_no_header = temp_df_no_header.iloc[:, :10]
                temp_df_no_header.columns = EXPECTED_COLUMNS
                df = temp_df_no_header
                used_sep = sep
                break

        except Exception as e:
            read_errors.append(f"sep={repr(sep)}: {str(e)}")

    if df is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No se pudo leer el CSV. "
                f"Errores: {' | '.join(read_errors)}"
            ),
        )

    calls: list[CallInput] = []

    for _, row in df.iterrows():
        calls.append(
            CallInput(
                id_conversacion=str(row.get("ID_CONVERSACION", "")).strip(),
                request_time=str(row.get("REQUEST_TIME", "")).strip() if pd.notna(row.get("REQUEST_TIME")) else None,
                rut_cliente=str(row.get("RUT_CLIENTE", "")).strip() if pd.notna(row.get("RUT_CLIENTE")) else None,
                phono_contacto=str(row.get("PHONO_CONTACTO", "")).strip() if pd.notna(row.get("PHONO_CONTACTO")) else None,
                fin_llamada=str(row.get("FIN_LLAMADA", "")).strip() if pd.notna(row.get("FIN_LLAMADA")) else None,
                marca_abandono=str(row.get("MARCA_ABANDONO", "")).strip() if pd.notna(row.get("MARCA_ABANDONO")) else None,
                marca_derivado=str(row.get("MARCA_DERIVADO", "")).strip() if pd.notna(row.get("MARCA_DERIVADO")) else None,
                cod_tipificacion=str(row.get("COD_TIPIFICACION", "")).strip() if pd.notna(row.get("COD_TIPIFICACION")) else None,
                ivr=str(row.get("IVR", "")).strip() if pd.notna(row.get("IVR")) else None,
                conversacion=str(row.get("CONVERSACION", "")).strip() if pd.notna(row.get("CONVERSACION")) else "",
            )
        )

    rows = [classifier_service.classify(call) for call in calls]
    csv_content = rows_to_csv(rows)

    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="resultado_amalia.csv"',
            "X-Detected-Separator": str(used_sep),
        },
    )
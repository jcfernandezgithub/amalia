from typing import List, Optional
from pydantic import BaseModel, Field


class DictionaryEntry(BaseModel):
    caso: str
    cod_tipo: int
    cod_subtipo: int
    tipo: str
    subtipo: str
    activo: bool = True
    tags: List[str] = Field(default_factory=list)


class CallInput(BaseModel):
    id_conversacion: str
    request_time: Optional[str] = None
    rut_cliente: Optional[str] = None
    phono_contacto: Optional[str] = None
    fin_llamada: Optional[str] = None
    marca_abandono: Optional[str] = None
    marca_derivado: Optional[str] = None
    cod_tipificacion: Optional[str] = None
    ivr: Optional[str] = None
    conversacion: str


class AnalyzeRequest(BaseModel):
    calls: List[CallInput]
    prompt: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash"


class AnalyzeRow(BaseModel):
    id_conversacion: str
    tipo: str
    subtipo: str
    cod_tipo: int
    cod_subtipo: int
    resolucion: str
    satisfaccion: str
    loop: int
    falla_ia: str
    compra_tarjeta: int
    compra_av_sav: int
    compra_seguro: int
    opcion_pago: int


class AnalyzeResponse(BaseModel):
    total: int
    rows: List[AnalyzeRow]
    csv: str

class GeminiClassificationResult(BaseModel):
    tipo: str
    subtipo: str
    cod_tipo: int
    cod_subtipo: int
    resolucion: str
    satisfaccion: str
    falla_ia: str
    compra_tarjeta: int
    compra_av_sav: int
    compra_seguro: int
    opcion_pago: int
    confidence: float

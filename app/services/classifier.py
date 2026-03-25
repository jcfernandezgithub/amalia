import json
import os
from typing import List, Optional
from google import genai
from google.genai import types
from app.models.schemas import AnalyzeRow, CallInput, DictionaryEntry
from app.services.transcript_parser import TranscriptParser
from pydantic import BaseModel

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


class GeminiClassifierService:
    def __init__(self, dictionary_entries: List[DictionaryEntry]) -> None:
        self.dictionary_entries = dictionary_entries
        self.parser = TranscriptParser()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def classify(
        self,
        call: CallInput,
        prompt_from_front: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AnalyzeRow:
        parsed = self.parser.parse(call.conversacion)
        raw_lower = parsed.raw_text.lower()
        customer_text = " ".join(parsed.customer_messages).lower()
        loop = self.parser.count_exact_repetition(parsed.customer_messages)

        hard_rule = self._apply_hard_rules(call, parsed.raw_text, customer_text)
        if hard_rule is not None:
            return hard_rule

        gemini_result = self._classify_with_gemini(
            call=call,
            parsed_text=parsed.raw_text,
            customer_text=customer_text,
            loop=loop,
            prompt_from_front=prompt_from_front,
            model=model or self.default_model,
        )

        return AnalyzeRow(
            id_conversacion=call.id_conversacion,
            tipo=gemini_result.tipo,
            subtipo=gemini_result.subtipo,
            cod_tipo=gemini_result.cod_tipo,
            cod_subtipo=gemini_result.cod_subtipo,
            resolucion=gemini_result.resolucion,
            satisfaccion=gemini_result.satisfaccion,
            loop=loop,
            falla_ia=gemini_result.falla_ia,
            compra_tarjeta=gemini_result.compra_tarjeta,
            compra_av_sav=gemini_result.compra_av_sav,
            compra_seguro=gemini_result.compra_seguro,
            opcion_pago=gemini_result.opcion_pago,
        )

    def _classify_with_gemini(
        self,
        call: CallInput,
        parsed_text: str,
        customer_text: str,
        loop: int,
        prompt_from_front: Optional[str],
        model: str,
    ) -> GeminiClassificationResult:
        dictionary_json = json.dumps(
            [entry.model_dump() for entry in self.dictionary_entries],
            ensure_ascii=False,
            indent=2,
        )

        base_instruction = f"""
Eres un clasificador experto de conversaciones de call center de Tarjeta Lider BCI.

Tu tarea es clasificar UNA conversación en una sola tipificación del diccionario entregado.
Debes elegir únicamente una categoría válida del diccionario.
No inventes códigos ni nombres nuevos.

Reglas:
1. Usa solo categorías del diccionario.
2. Prioriza el mensaje del cliente.
3. Si hay reclamo por cobro, diferencia, monto no coincidente o estado de cuenta distinto, prioriza cod_tipo=31 cod_subtipo=7 si aplica.
4. resolucion solo puede ser: "Sí" o "No".
5. satisfaccion solo puede ser: "Satisfecho", "Neutro" o "Enojado".
6. falla_ia solo puede ser:
   - "Sin_Error_IA"
   - "Error_Idioma"
   - "Falsa_Deteccion_OF"
   - "Logica_Circular"
   - "Respuesta_Inadecuada"
7. compra_tarjeta, compra_av_sav, compra_seguro, opcion_pago deben ser 0 o 1.
8. confidence debe ser un número entre 0 y 1.
9. Si no encuentras match claro, usa:
   tipo="Seguridad y Bloqueos"
   subtipo="Sin reconocimiento por Amalia"
   cod_tipo=31
   cod_subtipo=0

Diccionario:
{dictionary_json}
""".strip()

        extra_instruction = (prompt_from_front or "").strip()

        contents = f"""
INSTRUCCION_BASE:
{base_instruction}

INSTRUCCION_ADICIONAL_USUARIO:
{extra_instruction}

METADATA:
- id_conversacion: {call.id_conversacion}
- marca_abandono: {call.marca_abandono}
- marca_derivado: {call.marca_derivado}
- loop_detectado: {loop}

TEXTO_CLIENTE:
{customer_text}

CONVERSACION_COMPLETA:
{parsed_text}
""".strip()

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=GeminiClassificationResult,
            ),
        )

        parsed = response.parsed
        if parsed is None:
            raise ValueError("Gemini no devolvió una respuesta estructurada válida")

        return parsed

    def _apply_hard_rules(
        self, call: CallInput, raw_text: str, customer_text: str
    ) -> Optional[AnalyzeRow]:
        text = (raw_text or "").strip().lower()

        if (
            not text
            or len(text) < 15
            or (call.marca_abandono == "1" and len(customer_text.strip()) == 0)
        ):
            return AnalyzeRow(
                id_conversacion=call.id_conversacion,
                tipo="Amalia",
                subtipo="Llamada abandonada / Sin termino",
                cod_tipo=10,
                cod_subtipo=3,
                resolucion="No",
                satisfaccion="Neutro",
                loop=0,
                falla_ia="Sin_Error_IA",
                compra_tarjeta=0,
                compra_av_sav=0,
                compra_seguro=0,
                opcion_pago=0,
            )

        if self._contains_offensive_language(customer_text):
            return AnalyzeRow(
                id_conversacion=call.id_conversacion,
                tipo="Amalia",
                subtipo="Cliente Ofensivo",
                cod_tipo=10,
                cod_subtipo=4,
                resolucion="No",
                satisfaccion="Enojado",
                loop=0,
                falla_ia="Sin_Error_IA",
                compra_tarjeta=0,
                compra_av_sav=0,
                compra_seguro=0,
                opcion_pago=0,
            )

        return None

    def _contains_offensive_language(self, text: str) -> bool:
        offensive = [
            "huev",
            "idiota",
            "estúpido",
            "imbécil",
            "concha",
            "mierda",
            "puta",
            "weon",
            "culiao",
        ]
        return any(word in text for word in offensive)
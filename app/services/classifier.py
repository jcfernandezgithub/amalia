import re
from typing import List, Optional
from app.models.schemas import AnalyzeRow, CallInput, DictionaryEntry
from app.services.transcript_parser import TranscriptParser


class ClassifierService:
    def __init__(self, dictionary_entries: List[DictionaryEntry]) -> None:
        self.dictionary_entries = dictionary_entries
        self.parser = TranscriptParser()

    def classify(self, call: CallInput) -> AnalyzeRow:
        parsed = self.parser.parse(call.conversacion)
        raw_lower = parsed.raw_text.lower()
        customer_text = " ".join(parsed.customer_messages).lower()
        loop = self.parser.count_exact_repetition(parsed.customer_messages)

        hard_rule = self._apply_hard_rules(call, parsed.raw_text, customer_text)
        if hard_rule is not None:
            return hard_rule

        best_match = self._match_dictionary(customer_text, raw_lower)

        resolucion = self._resolve_resolution(call, raw_lower)
        satisfaccion = self._resolve_satisfaction(customer_text, raw_lower)
        falla_ia = self._resolve_ai_failure(customer_text, raw_lower)
        compra_tarjeta, compra_av_sav, compra_seguro, opcion_pago = (
            self._commercial_flags(customer_text)
        )

        if best_match is None:
            best_match = DictionaryEntry(
                caso="Sin reconocimiento por Amalia",
                cod_tipo=31,
                cod_subtipo=0,
                tipo="Seguridad y Bloqueos",
                subtipo="Sin reconocimiento por Amalia",
                activo=True,
                tags=[],
            )

        return AnalyzeRow(
            id_conversacion=call.id_conversacion,
            tipo=best_match.tipo,
            subtipo=best_match.subtipo,
            cod_tipo=best_match.cod_tipo,
            cod_subtipo=best_match.cod_subtipo,
            resolucion=resolucion,
            satisfaccion=satisfaccion,
            loop=loop,
            falla_ia=falla_ia,
            compra_tarjeta=compra_tarjeta,
            compra_av_sav=compra_av_sav,
            compra_seguro=compra_seguro,
            opcion_pago=opcion_pago,
        )

    def _apply_hard_rules(
        self, call: CallInput, raw_text: str, customer_text: str
    ) -> Optional[AnalyzeRow]:
        text = (raw_text or "").strip().lower()

        if (
            not text
            or len(text) < 15
            or (call.marca_abandono == "1" and len(customer_text.strip()) == 0)
        ):
            return self._build_row(
                call.id_conversacion,
                "Amalia",
                "Llamada abandonada / Sin termino",
                10,
                3,
                "No",
                "Neutro",
                0,
                "Sin_Error_IA",
                0,
                0,
                0,
                0,
            )

        if self._contains_offensive_language(customer_text):
            return self._build_row(
                call.id_conversacion,
                "Amalia",
                "Cliente Ofensivo",
                10,
                4,
                "No",
                "Enojado",
                0,
                "Sin_Error_IA",
                0,
                0,
                0,
                0,
            )

        return None

    def _match_dictionary(
        self, customer_text: str, raw_lower: str
    ) -> Optional[DictionaryEntry]:
        scored: List[tuple[int, DictionaryEntry]] = []

        for entry in self.dictionary_entries:
            score = 0
            case_text = entry.caso.lower()
            subtype_text = entry.subtipo.lower()
            type_text = entry.tipo.lower()

            for token in self._keywords_from_text(
                case_text, subtype_text, type_text, entry.tags
            ):
                if token and token in customer_text:
                    score += 3

            if entry.cod_tipo == 31:
                for token in [
                    "problema",
                    "error",
                    "no puedo",
                    "diferencia",
                    "cobran",
                    "cobro",
                    "reclamo",
                ]:
                    if token in customer_text:
                        score += 1

            if entry.cod_tipo == 12 and any(
                x in customer_text
                for x in ["cuánto debo", "deuda", "facturado", "cupo"]
            ):
                score += 2

            if score > 0:
                scored.append((score, entry))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0][1]

        # regla simple de prioridad para reclamos financieros
        if any(
            x in customer_text
            for x in [
                "cobran más",
                "me cobran más",
                "diferencia",
                "no coincide",
                "sale que debo",
            ]
        ):
            manual = self._find_exact(31, 7)
            if manual:
                return manual

        return top

    def _resolve_resolution(self, call: CallInput, raw_lower: str) -> str:
        if call.marca_derivado:
            return "No"
        if (
            "transfer" in raw_lower
            or "especialista" in raw_lower
            or "ejecutivo" in raw_lower
        ):
            return "No"
        if "[sin respuesta]" in raw_lower:
            return "No"
        if (
            "gracias por llamar" in raw_lower
            and "encuesta de satisfacción" in raw_lower
            and call.marca_abandono == "1"
        ):
            return "No"
        return "Sí"

    def _resolve_satisfaction(self, customer_text: str, raw_lower: str) -> str:
        angry_markers = [
            "robot",
            "me cobran más",
            "no sirve",
            "quiero hablar con un ejecutivo",
            "ejecutivo",
            "reclamo",
            "insólito",
            "molesto",
            "enojado",
        ]
        satisfied_markers = ["gracias", "perfecto", "ok gracias", "muchas gracias"]

        if any(x in customer_text for x in angry_markers):
            return "Enojado"
        if any(x in customer_text for x in satisfied_markers):
            return "Satisfecho"
        return "Neutro"

    def _resolve_ai_failure(self, customer_text: str, raw_lower: str) -> str:
        if self._has_language_mismatch(customer_text, raw_lower):
            return "Error_Idioma"
        if self._has_false_offensive_detection(raw_lower):
            return "Falsa_Deteccion_OF"
        if self._has_circular_logic(raw_lower):
            return "Logica_Circular"
        if self._has_inadequate_response(customer_text, raw_lower):
            return "Respuesta_Inadecuada"
        return "Sin_Error_IA"

    def _commercial_flags(self, customer_text: str) -> tuple[int, int, int, int]:
        compra_tarjeta = (
            1
            if any(
                x in customer_text
                for x in [
                    "quiero sacar la tarjeta",
                    "quiero pedir la tarjeta",
                    "quiero una tarjeta adicional",
                ]
            )
            else 0
        )

        compra_av_sav = (
            1
            if any(
                x in customer_text
                for x in [
                    "quiero un avance",
                    "quiero sacar un avance",
                    "quiero pedir un super avance",
                    "necesito platita",
                    "quiero un super avance",
                ]
            )
            else 0
        )

        compra_seguro = (
            1
            if any(
                x in customer_text
                for x in [
                    "quiero contratar un seguro",
                    "quiero un seguro",
                    "quiero pedir un seguro",
                ]
            )
            else 0
        )

        opcion_pago = (
            1
            if any(
                x in customer_text
                for x in ["cuota flexible", "pago liviano", "refi", "rene"]
            )
            else 0
        )

        return compra_tarjeta, compra_av_sav, compra_seguro, opcion_pago

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

    def _has_language_mismatch(self, customer_text: str, raw_lower: str) -> bool:
        return bool(customer_text.strip()) and (
            ("hello" in raw_lower or "how can i help" in raw_lower)
            and not any(x in customer_text for x in ["hello", "hi"])
        )

    def _has_false_offensive_detection(self, raw_lower: str) -> bool:
        return "ofensivo" in raw_lower and not any(
            x in raw_lower for x in ["idiota", "mierda", "puta", "imbécil"]
        )

    def _has_circular_logic(self, raw_lower: str) -> bool:
        patterns = [
            "¿en qué puedo ayudarte hoy?",
            "¿cuál es tu consulta?",
            "estoy aquí para ayudarte",
        ]
        count = sum(raw_lower.count(p) for p in patterns)
        return count >= 3

    def _has_inadequate_response(self, customer_text: str, raw_lower: str) -> bool:
        sensitive = [
            "falleció",
            "fallecimiento",
            "adulto mayor",
            "no tengo internet",
            "no sé usar la app",
        ]
        if any(x in customer_text for x in sensitive):
            return True
        if "no tengo información específica" in raw_lower:
            return True
        if "puedes consultar en la app" in raw_lower and any(
            x in customer_text for x in ["diferencia", "me cobran más", "reclamo"]
        ):
            return True
        return False

    def _keywords_from_text(self, *chunks) -> List[str]:
        words: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, list):
                for item in chunk:
                    words.extend(self._tokenize(item))
            else:
                words.extend(self._tokenize(chunk))
        return list(set(w for w in words if len(w) >= 4))

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-záéíóúüñ0-9\s/-]", " ", text)
        return [w.strip() for w in text.split() if w.strip()]

    def _find_exact(self, cod_tipo: int, cod_subtipo: int) -> Optional[DictionaryEntry]:
        for entry in self.dictionary_entries:
            if entry.cod_tipo == cod_tipo and entry.cod_subtipo == cod_subtipo:
                return entry
        return None

    def _build_row(
        self,
        id_conversacion: str,
        tipo: str,
        subtipo: str,
        cod_tipo: int,
        cod_subtipo: int,
        resolucion: str,
        satisfaccion: str,
        loop: int,
        falla_ia: str,
        compra_tarjeta: int,
        compra_av_sav: int,
        compra_seguro: int,
        opcion_pago: int,
    ) -> AnalyzeRow:
        return AnalyzeRow(
            id_conversacion=id_conversacion,
            tipo=tipo,
            subtipo=subtipo,
            cod_tipo=cod_tipo,
            cod_subtipo=cod_subtipo,
            resolucion=resolucion,
            satisfaccion=satisfaccion,
            loop=loop,
            falla_ia=falla_ia,
            compra_tarjeta=compra_tarjeta,
            compra_av_sav=compra_av_sav,
            compra_seguro=compra_seguro,
            opcion_pago=opcion_pago,
        )

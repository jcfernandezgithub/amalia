import json
from pathlib import Path
from typing import List
from app.models.schemas import DictionaryEntry


class DictionaryService:
    def __init__(self, file_path: str = "app/data/tipificaciones.json") -> None:
        self.file_path = Path(file_path)
        self.entries: List[DictionaryEntry] = self._load()

    def _load(self) -> List[DictionaryEntry]:
        with self.file_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        entries = [DictionaryEntry(**item) for item in raw]
        return [e for e in entries if e.activo]

    def all(self) -> List[DictionaryEntry]:
        return self.entries

    def find_by_code(self, cod_tipo: int, cod_subtipo: int) -> DictionaryEntry | None:
        for entry in self.entries:
            if entry.cod_tipo == cod_tipo and entry.cod_subtipo == cod_subtipo:
                return entry
        return None

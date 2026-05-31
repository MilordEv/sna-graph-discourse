"""
Загрузчик датасета «Сайт Русская Истина».

Читает xlsx с колонками: дата, автор, название, ссылка, о чём, теги.
Сохраняет data/raw/russkaya_istina/documents.json в формате, совместимом
с конструктором дискурс-графа.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

# Слова-паразиты из вёрстки, которые не несут смысла
NOISE_RE = re.compile(
    r"\b(автор|редакция|главная|новости|теги|фото|опубликовано|читать далее)\b",
    re.I,
)


def _read_shared_strings(z: zipfile.ZipFile) -> list[str]:
    with z.open("xl/sharedStrings.xml") as f:
        tree = ET.parse(f)
    return [
        "".join(t.text or "" for t in si.iter(f"{{{NS}}}t"))
        for si in tree.findall(f".//{{{NS}}}si")
    ]


def _read_sheet(z: zipfile.ZipFile, name: str, strings: list[str]) -> list[dict]:
    with z.open(f"xl/worksheets/{name}") as f:
        ws = ET.parse(f)
    rows = []
    for row in ws.findall(f".//{{{NS}}}row"):
        cells: dict[str, str] = {}
        for c in row.findall(f"{{{NS}}}c"):
            ref = c.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            t = c.get("t", "")
            v_el = c.find(f"{{{NS}}}v")
            if v_el is None:
                continue
            cells[col] = strings[int(v_el.text)] if t == "s" else (v_el.text or "")
        rows.append(cells)
    return rows


def _excel_date(serial: str | None) -> str:
    if not serial:
        return ""
    try:
        n = int(float(serial))
        if n > 59:
            n -= 1  # пропустить несуществующую дату 1900-02-29
        return (date(1899, 12, 31) + timedelta(days=n)).isoformat()
    except Exception:
        return str(serial)


def _parse_tags(raw: str) -> list[str]:
    if not raw or raw.strip() == "1":
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def load_xlsx(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as z:
        strings = _read_shared_strings(z)
        rows = _read_sheet(z, "sheet1.xml", strings)

    if not rows:
        return []

    header = rows[0]
    # Определяем маппинг колонок по значениям заголовка
    col_map = {v.strip().lower(): k for k, v in header.items()}
    data_rows = rows[1:]

    docs: list[dict] = []
    for i, row in enumerate(data_rows):
        def cell(name: str) -> str:
            return row.get(col_map.get(name, ""), "").strip()

        title = cell("название")
        author = cell("автор")
        summary = cell("о чем")
        url = cell("ссылка")
        tags = _parse_tags(cell("теги"))
        date_str = _excel_date(row.get(col_map.get("дата", "A"), ""))

        if not title and not summary:
            continue

        # Формируем текст: название + автор + аннотация
        parts = [p for p in [title, author, summary] if p]
        text = ". ".join(parts)

        # Теги — прямые ключевые слова для YAKE-экстрактора
        keywords = list(dict.fromkeys(tags))  # дедупликация с сохранением порядка

        docs.append({
            "doc_id": f"ri_{i + 1:04d}",
            "source": "russkaya_istina",
            "url": url,
            "date": date_str,
            "title": title,
            "text": text,
            "authors": [author] if author else [],
            "tags": tags,
            "keywords": keywords,
            "entities": [],
            "voice_type": "article",
        })

    return docs


def main() -> None:
    src = Path("Сайт_Русская_Истина_Статьи_по_тегам_истина_правда_ложь.xlsx")
    if not src.exists():
        raise FileNotFoundError(f"Не найден файл: {src}")

    out_dir = Path("data/raw/russkaya_istina")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "documents.json"

    docs = load_xlsx(src)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Сохранено {len(docs)} документов → {out}")
    print()
    lengths = [len(d["text"]) for d in docs]
    print(f"Длина текста: min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")
    print()
    all_tags: list[str] = []
    for d in docs:
        all_tags.extend(d["tags"])
    from collections import Counter
    print("Топ тегов:", Counter(all_tags).most_common(10))
    print()
    print("Примеры:")
    for d in docs[:3]:
        print(f"  [{d['doc_id']}] {d['title'][:60]}")
        print(f"           tags={d['tags']} | len(text)={len(d['text'])}")


if __name__ == "__main__":
    main()

"""RAG Q&A — FastAPI. Загрузи документ (текст или PDF) → задавай вопросы →
ответы с опорой на источник. Кнопка «Эвал» гоняет golden-набор и показывает метрики.
"""

from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import rag
import evaluate

app = FastAPI(title="RAG Q&A")
HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
SAMPLE = (Path(__file__).parent / "eval" / "sample_doc.txt").read_text(encoding="utf-8")


class IngestReq(BaseModel):
    text: str


class AskReq(BaseModel):
    question: str
    k: int = 4


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/sample")
def sample():
    return {"text": SAMPLE}


@app.post("/api/ingest")
def ingest(r: IngestReq):
    if not r.text.strip():
        return {"error": "Пустой документ."}
    return {"chunks": rag.ingest(r.text)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            return {"error": f"Не смог прочитать PDF: {e}"}
    else:
        text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        return {"error": "В файле не нашлось текста."}
    return {"chunks": rag.ingest(text), "chars": len(text)}


@app.post("/api/ask")
def ask(r: AskReq):
    if not r.question.strip():
        return {"error": "Пустой вопрос."}
    if not rag.stats()["indexed"]:
        return {"error": "Сначала загрузи документ."}
    return rag.answer(r.question, max(1, min(r.k, 8)))


@app.post("/api/eval")
def run_eval():
    return evaluate.run()

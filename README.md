# RAG Q&A 📄🔎 — вопросы к документу с опорой на источник

Загружаешь документ (текст или PDF) → задаёшь вопросы → модель отвечает **только по тексту**,
с указанием, из каких фрагментов взят ответ. Плюс встроенный **мини-эвал**: golden-набор
вопросов + LLM-судья считают качество (retrieval hit-rate и accuracy).

**Стек:** Python · FastAPI · **fastembed** (локальные эмбеддинги, без API-ключей) ·
косинусный поиск (numpy) · LLM (Groq, бесплатно) · pypdf.

## Почему это важно для AI-инженера
- **RAG честно, без чёрного ящика:** видно чанкинг, эмбеддинги, ретривал и сборку контекста.
- **Evaluation** — то, что спрашивают на собесах и чего избегают джуны: качество измеряется
  (`retrieval_hit_rate`, `accuracy` через LLM-as-judge на golden-наборе), а не «на глаз».

## Метрики на демо-документе
`python evaluate.py` → **retrieval hit-rate 100% · accuracy 100%** (7 вопросов, gpt-oss-20b).

## Запуск локально
1. Бесплатный ключ Groq: **console.groq.com** → `copy .env.example .env`, впиши `GROQ_API_KEY`.
2. ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app:app --reload
   ```
3. http://127.0.0.1:8000 — загрузи пример, задай вопрос, нажми «Запустить эвал».

## Как устроено
- `rag.py` — чанкинг → эмбеддинги (fastembed) → косинусный поиск → ответ с цитатами.
- `evaluate.py` — прогон golden-набора (`eval/golden.json`) по демо-документу + LLM-судья.
- `app.py` — FastAPI: `/api/ingest`, `/api/upload`, `/api/ask`, `/api/eval`.
- `static/index.html` — интерфейс.

## Деплой
`Dockerfile` готов (напр. Amvera). `GROQ_API_KEY` — переменной окружения, порт 8000.
Эмбеддинги локальные, поэтому это бэкенд-сервис (не статический сайт).

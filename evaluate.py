"""Мини-эвал RAG: прогоняет golden-набор по демо-документу и считает метрики.

- retrieval hit-rate: попал ли нужный факт (keyword) хотя бы в один найденный чанк;
- accuracy (LLM-as-judge): строгая модель сверяет ответ с эталоном.
Это то, что отличает senior-подход: качество измеряется, а не «на глаз».
"""

import json
import re
from pathlib import Path

import rag
import llm

BASE = Path(__file__).parent / "eval"


def _judge(question, expected, answer):
    system = ("You are a strict grader. Given a question, a reference answer and a candidate "
              "answer, decide if the candidate is correct (contains the key facts of the "
              'reference). Reply STRICT JSON: {"correct": true|false, "note": "short"}')
    user = f"QUESTION: {question}\nREFERENCE: {expected}\nCANDIDATE: {answer}"
    try:
        reply, _ = llm.chat([{"role": "system", "content": system},
                             {"role": "user", "content": user}], max_tokens=200, temperature=0)
        m = re.search(r"\{.*\}", reply, re.S)
        return json.loads(m.group(0)) if m else {"correct": False, "note": "parse error"}
    except Exception as e:
        return {"correct": False, "note": str(e)}


def run(k: int = 4):
    doc = (BASE / "sample_doc.txt").read_text(encoding="utf-8")
    golden = json.loads((BASE / "golden.json").read_text(encoding="utf-8"))
    n_chunks = rag.ingest(doc)

    rows, hits, correct = [], 0, 0
    for item in golden:
        got = rag.retrieve(item["q"], k)
        retrieved = " ".join(t for _, _, t in got).lower()
        hit = item["keyword"].lower() in retrieved
        res = rag.answer(item["q"], k)
        verdict = _judge(item["q"], item["expected"], res["answer"])
        ok = bool(verdict.get("correct"))
        hits += hit
        correct += ok
        rows.append({"q": item["q"], "answer": res["answer"], "retrieval_hit": hit,
                     "correct": ok, "note": verdict.get("note", "")})

    total = len(golden)
    return {
        "chunks": n_chunks,
        "questions": total,
        "retrieval_hit_rate": round(hits / total, 3),
        "accuracy": round(correct / total, 3),
        "rows": rows,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    r = run()
    print(f"Чанков: {r['chunks']} | Вопросов: {r['questions']}")
    print(f"Retrieval hit-rate: {r['retrieval_hit_rate']} | Accuracy (LLM-judge): {r['accuracy']}\n")
    for row in r["rows"]:
        mark = "✅" if row["correct"] else "❌"
        print(f"{mark} {row['q']}")
        print(f"   → {row['answer'][:160]}")

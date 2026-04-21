from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "scenes.json"

app = FastAPI(title="Civil Aviation Emergency KG System 7.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class AskRequest(BaseModel):
    question: str
    top_k: int = 10
    final_n: int = 3


@lru_cache(maxsize=1)
def load_data() -> dict[str, Any]:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_rerank_model():
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def simple_similarity(text_a: str, text_b: str) -> float:
    set_a = set(text_a)
    set_b = set(text_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def build_scene_vectors() -> list[dict[str, Any]]:
    data = load_data()["scenes"]
    model = get_sentence_model()
    rows: list[dict[str, Any]] = []
    if model:
        corpus = []
        for scene in data:
            text = " ".join([scene["standard_term"], scene["category"], scene["summary"], " ".join(scene["aliases"])])
            corpus.append(text)
        embeddings = model.encode(corpus, normalize_embeddings=True)
        for scene, emb in zip(data, embeddings):
            rows.append({"scene": scene, "vector": emb})
    else:
        for scene in data:
            rows.append({"scene": scene, "vector": None})
    return rows


@lru_cache(maxsize=1)
def scene_vectors_cached() -> tuple[dict[str, Any], ...]:
    return tuple(build_scene_vectors())


def match_scene(question: str) -> dict[str, Any]:
    model = get_sentence_model()
    best_score = -1.0
    best_scene = None
    if model:
        qvec = model.encode([question], normalize_embeddings=True)[0]
        for row in scene_vectors_cached():
            score = cosine_similarity(qvec, row["vector"])
            if score > best_score:
                best_score = score
                best_scene = row["scene"]
    else:
        for row in scene_vectors_cached():
            scene = row["scene"]
            score = simple_similarity(question, scene["standard_term"] + "".join(scene["aliases"]))
            if score > best_score:
                best_score = score
                best_scene = scene

    aliases = best_scene["aliases"] if best_scene else []
    matched_aliases = [a for a in aliases if a in question]
    standard_terms = [best_scene["standard_term"]] if best_scene else []
    normalized_question = question
    if best_scene:
        normalized_question = f"{question}（标准场景：{best_scene['standard_term']}）"

    return {
        "scene": best_scene,
        "score": round(best_score, 4),
        "matched_aliases": matched_aliases,
        "standard_terms": standard_terms,
        "normalized_question": normalized_question,
    }


def retrieve_documents(question: str, scene: dict[str, Any], top_k: int, final_n: int) -> list[dict[str, Any]]:
    docs = list(scene["documents"])
    queries = [question, scene["standard_term"], scene["summary"]]
    query_text = " ".join(queries)
    embed_model = get_sentence_model()
    rerank_model = get_rerank_model()

    scored = []
    if embed_model:
        doc_vectors = embed_model.encode([d["text"] for d in docs], normalize_embeddings=True)
        qvec = embed_model.encode([query_text], normalize_embeddings=True)[0]
        for d, dv in zip(docs, doc_vectors):
            score = cosine_similarity(qvec, dv)
            scored.append({**d, "vector_score": round(score, 4)})
    else:
        for d in docs:
            score = simple_similarity(query_text, d["text"])
            scored.append({**d, "vector_score": round(score, 4)})

    scored.sort(key=lambda x: x["vector_score"], reverse=True)
    candidates = scored[: max(1, min(top_k, len(scored)))]

    if rerank_model:
        pairs = [[query_text, c["text"]] for c in candidates]
        rerank_scores = rerank_model.predict(pairs)
        for c, rs in zip(candidates, rerank_scores):
            c["rerank_score"] = round(float(rs), 4)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        for c in candidates:
            c["rerank_score"] = c["vector_score"]

    return candidates[: max(1, min(final_n, len(candidates)))]


def build_subgraph(scene: dict[str, Any], evidences: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    all_nodes = scene["graph"]["nodes"]
    all_edges = scene["graph"]["edges"]
    text_join = " ".join([e["title"] + e["text"] for e in evidences]) + scene["standard_term"]

    selected_ids = set()
    seed_terms = []
    for node in all_nodes:
        if node["label"] in text_join or node["label"] == scene["standard_term"]:
            selected_ids.add(node["id"])
            seed_terms.append(node["label"])

    # always include central event and its first-hop neighbors
    central = None
    for node in all_nodes:
        if node["label"] == scene["standard_term"]:
            central = node["id"]
            selected_ids.add(node["id"])
            break

    for edge in all_edges:
        if edge["from"] in selected_ids or edge["to"] in selected_ids or edge["from"] == central:
            selected_ids.add(edge["from"])
            selected_ids.add(edge["to"])

    nodes = [n for n in all_nodes if n["id"] in selected_ids]
    edges = [e for e in all_edges if e["from"] in selected_ids and e["to"] in selected_ids]
    if not seed_terms:
        seed_terms = [scene["standard_term"]]
    return nodes, edges, list(dict.fromkeys(seed_terms))


def compose_answer(scene: dict[str, Any], evidences: list[dict[str, Any]]) -> str:
    steps = scene["steps"]
    notes = scene["notes"]
    rules = scene["rules"]

    lines = []
    lines.append(f"针对“{scene['standard_term']}”，系统建议优先从识别场景、快速报告、实施处置和运行决策四个方面开展应急响应。")
    lines.append("")
    lines.append("一、建议的处置流程")
    for idx, s in enumerate(steps, 1):
        lines.append(f"{idx}. {s}")
    lines.append("")
    lines.append("二、风险提示")
    for idx, r in enumerate(scene["risks"], 1):
        lines.append(f"{idx}. {r}")
    lines.append("")
    lines.append("三、执行注意事项")
    for idx, n in enumerate(notes, 1):
        lines.append(f"{idx}. {n}")
    lines.append("")
    lines.append("四、依据说明")
    lines.append(f"可重点参考：{'、'.join(rules)}。")
    if evidences:
        lines.append(f"本次回答综合了 {len(evidences)} 条高相关证据文本，用于支撑处置建议与图谱展开。")
    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask")
def ask_api(payload: AskRequest):
    matched = match_scene(payload.question)
    scene = matched["scene"]
    if not scene:
        return {
            "normalized_question": payload.question,
            "mapped_terms": [],
            "standard_terms": [],
            "graph_summary": "未识别到匹配场景。",
            "graph_seed_terms": [],
            "answer": "当前未识别到明确的民航应急场景，建议补充更具体的问题描述，例如事件位置、飞行阶段或旅客症状。",
            "evidences": [],
            "graph_nodes": [],
            "graph_edges": [],
        }

    evidences = retrieve_documents(payload.question, scene, payload.top_k, payload.final_n)
    graph_nodes, graph_edges, seed_terms = build_subgraph(scene, evidences)
    answer = compose_answer(scene, evidences)
    graph_summary = f"已围绕“{scene['standard_term']}”生成问题相关子图，覆盖事件、步骤、岗位、设备与规则依据。"

    mapped_terms = matched["matched_aliases"] or [scene["standard_term"]]

    return {
        "normalized_question": matched["normalized_question"],
        "mapped_terms": mapped_terms,
        "standard_terms": matched["standard_terms"],
        "graph_summary": graph_summary,
        "graph_seed_terms": seed_terms,
        "answer": answer,
        "evidences": evidences,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
    }

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


# ✅ 已关闭 embedding 模型（避免 Render 崩溃）
@lru_cache(maxsize=1)
def get_sentence_model():
    return None


# ✅ 已关闭 rerank 模型（避免 Render 崩溃）
@lru_cache(maxsize=1)
def get_rerank_model():
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
    rows: list[dict[str, Any]] = []
    for scene in data:
        rows.append({"scene": scene, "vector": None})
    return rows


@lru_cache(maxsize=1)
def scene_vectors_cached() -> tuple[dict[str, Any], ...]:
    return tuple(build_scene_vectors())


def match_scene(question: str) -> dict[str, Any]:
    best_score = -1.0
    best_scene = None

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

    scored = []
    for d in docs:
        score = simple_similarity(query_text, d["text"])
        scored.append({**d, "vector_score": round(score, 4), "rerank_score": round(score, 4)})

    scored.sort(key=lambda x: x["vector_score"], reverse=True)
    return scored[: max(1, min(final_n, len(scored)))]


def build_subgraph(scene: dict[str, Any], evidences: list[dict[str, Any]]):
    all_nodes = scene["graph"]["nodes"]
    all_edges = scene["graph"]["edges"]
    return all_nodes, all_edges, [scene["standard_term"]]


def compose_answer(scene: dict[str, Any], evidences: list[dict[str, Any]]) -> str:
    return f"已识别场景：{scene['standard_term']}，系统已生成基础应急处置建议（简化模式运行）。"


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
            "answer": "未识别问题",
            "evidences": [],
            "graph_nodes": [],
            "graph_edges": [],
        }

    evidences = retrieve_documents(payload.question, scene, payload.top_k, payload.final_n)
    graph_nodes, graph_edges, seed_terms = build_subgraph(scene, evidences)
    answer = compose_answer(scene, evidences)

    return {
        "normalized_question": matched["normalized_question"],
        "mapped_terms": matched["matched_aliases"],
        "standard_terms": matched["standard_terms"],
        "graph_summary": "已生成简化知识图谱",
        "graph_seed_terms": seed_terms,
        "answer": answer,
        "evidences": evidences,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
    }

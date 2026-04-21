from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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


def simple_similarity(text_a: str, text_b: str) -> float:
    set_a = set(str(text_a))
    set_b = set(str(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def keyword_bonus(question: str, scene: dict[str, Any]) -> float:
    bonus = 0.0
    standard_term = scene.get("standard_term", "")
    aliases = scene.get("aliases", [])
    category = scene.get("category", "")
    summary = scene.get("summary", "")

    if standard_term and standard_term in question:
        bonus += 0.35

    for alias in aliases:
        if alias and alias in question:
            bonus += 0.18

    for text in [category, summary]:
        if text:
            overlap = len(set(question) & set(text))
            bonus += min(overlap * 0.005, 0.08)

    return bonus


def match_scene(question: str) -> dict[str, Any]:
    data = load_data().get("scenes", [])
    best_score = -1.0
    best_scene = None

    for scene in data:
        compare_text = (
            scene.get("standard_term", "")
            + "".join(scene.get("aliases", []))
            + scene.get("summary", "")
            + scene.get("category", "")
        )
        score = simple_similarity(question, compare_text) + keyword_bonus(question, scene)
        if score > best_score:
            best_score = score
            best_scene = scene

    if not best_scene:
        return {
            "scene": None,
            "score": 0.0,
            "matched_aliases": [],
            "standard_terms": [],
            "normalized_question": question,
        }

    aliases = best_scene.get("aliases", [])
    matched_aliases = [a for a in aliases if a in question]
    standard_terms = [best_scene.get("standard_term", "")]
    normalized_question = f"{question}（标准场景：{best_scene.get('standard_term', '')}）"

    return {
        "scene": best_scene,
        "score": round(best_score, 4),
        "matched_aliases": matched_aliases,
        "standard_terms": standard_terms,
        "normalized_question": normalized_question,
    }


def retrieve_documents(
    question: str, scene: dict[str, Any], top_k: int, final_n: int
) -> list[dict[str, Any]]:
    docs = list(scene.get("documents", []))
    query_text = " ".join(
        [
            question,
            scene.get("standard_term", ""),
            scene.get("summary", ""),
            " ".join(scene.get("aliases", [])),
        ]
    )

    scored = []
    for d in docs:
        text = d.get("text", "")
        title = d.get("title", "")
        source = d.get("source", "")
        mix_text = f"{title} {text} {source}"
        score = simple_similarity(query_text, mix_text)
        scored.append(
            {
                **d,
                "vector_score": round(score, 4),
                "rerank_score": round(score, 4),
            }
        )

    scored.sort(key=lambda x: x["vector_score"], reverse=True)

    if not scored:
        return []

    top_k = max(1, min(top_k, len(scored)))
    final_n = max(1, min(final_n, len(scored)))
    candidates = scored[:top_k]
    return candidates[:final_n]


def build_subgraph(
    scene: dict[str, Any], evidences: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    all_nodes = scene.get("graph", {}).get("nodes", [])
    all_edges = scene.get("graph", {}).get("edges", [])
    standard_term = scene.get("standard_term", "")

    text_join = " ".join(
        [str(e.get("title", "")) + str(e.get("text", "")) for e in evidences]
    ) + standard_term

    selected_ids = set()
    seed_terms = []

    for node in all_nodes:
        label = node.get("label", "")
        if label and (label in text_join or label == standard_term):
            selected_ids.add(node.get("id"))
            seed_terms.append(label)

    central = None
    for node in all_nodes:
        if node.get("label") == standard_term:
            central = node.get("id")
            selected_ids.add(node.get("id"))
            break

    for edge in all_edges:
        if (
            edge.get("from") in selected_ids
            or edge.get("to") in selected_ids
            or edge.get("from") == central
        ):
            selected_ids.add(edge.get("from"))
            selected_ids.add(edge.get("to"))

    nodes = [n for n in all_nodes if n.get("id") in selected_ids]
    edges = [
        e
        for e in all_edges
        if e.get("from") in selected_ids and e.get("to") in selected_ids
    ]

    if not nodes:
        nodes = all_nodes
    if not edges:
        edges = all_edges
    if not seed_terms:
        seed_terms = [standard_term] if standard_term else []

    seed_terms = list(dict.fromkeys(seed_terms))
    return nodes, edges, seed_terms


def compose_answer(scene: dict[str, Any], evidences: list[dict[str, Any]]) -> str:
    steps = scene.get("steps", [])
    notes = scene.get("notes", [])
    rules = scene.get("rules", [])
    risks = scene.get("risks", [])
    standard_term = scene.get("standard_term", "该场景")

    lines = []
    lines.append(f"针对“{standard_term}”，系统建议优先从识别场景、快速报告、实施处置和运行决策四个方面开展应急响应。")
    lines.append("")
    lines.append("一、建议的处置流程")
    if steps:
        for idx, s in enumerate(steps, 1):
            lines.append(f"{idx}. {s}")
    else:
        lines.append("1. 立即识别事件性质并确认影响范围。")
        lines.append("2. 按程序进行信息报告和协同处置。")
        lines.append("3. 结合现场情况落实控制措施并持续监控。")

    lines.append("")
    lines.append("二、风险提示")
    if risks:
        for idx, r in enumerate(risks, 1):
            lines.append(f"{idx}. {r}")
    else:
        lines.append("1. 需关注事件升级、次生风险及运行中断影响。")

    lines.append("")
    lines.append("三、执行注意事项")
    if notes:
        for idx, n in enumerate(notes, 1):
            lines.append(f"{idx}. {n}")
    else:
        lines.append("1. 处置过程中应保持信息准确、沟通顺畅和责任明确。")

    lines.append("")
    lines.append("四、依据说明")
    if rules:
        lines.append(f"可重点参考：{'、'.join(rules)}。")
    else:
        lines.append("可结合现行应急预案、运行手册和岗位程序执行。")

    if evidences:
        lines.append(f"本次回答综合了 {len(evidences)} 条高相关证据文本，用于支撑处置建议与图谱展开。")

    return "\n".join(lines)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ask")
def ask_api(payload: AskRequest):
    try:
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

        evidences = retrieve_documents(
            payload.question, scene, payload.top_k, payload.final_n
        )
        graph_nodes, graph_edges, seed_terms = build_subgraph(scene, evidences)
        answer = compose_answer(scene, evidences)
        graph_summary = f"已围绕“{scene.get('standard_term', '')}”生成问题相关子图，覆盖事件、步骤、岗位、设备与规则依据。"
        mapped_terms = matched["matched_aliases"] or [scene.get("standard_term", "")]

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

    except Exception as e:
        return {
            "normalized_question": payload.question,
            "mapped_terms": [],
            "standard_terms": [],
            "graph_summary": "系统运行异常。",
            "graph_seed_terms": [],
            "answer": f"系统处理过程中出现错误：{str(e)}",
            "evidences": [],
            "graph_nodes": [],
            "graph_edges": [],
        }

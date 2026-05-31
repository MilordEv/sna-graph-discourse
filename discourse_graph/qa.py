"""
LLM Q&A поверх дискурс-графа.

Поддерживает три стратегии ретривала: walk, community, lightrag.
Работает с любым OpenAI-совместимым API (DeepSeek, OpenAI, Gemini прокси).

Пример использования:
    from openai import OpenAI
    from discourse_graph.qa import DiscourseQA
    import networkx as nx

    G = nx.read_graphml("data/graphs/russkaya_istina/constructor/discourse/discourse_graph.graphml")
    client = OpenAI(api_key="sk-or-...", base_url="https://openrouter.ai/api/v1")
    qa = DiscourseQA(G, client, model="deepseek/deepseek-chat")
    answer = qa.answer("Какие концепты противопоставляются истине?")
    print(answer)
"""
from __future__ import annotations

from typing import Callable

import networkx as nx

from discourse_graph.retrieval import retrieve_community, retrieve_lightrag, retrieve_walk

SYSTEM_PROMPT = """Ты — аналитик дискурса. Тебе предоставлен контекст из дискурс-графа:
граф отражает семантическую связанность концептов в текстовом корпусе.
Узлы — концепты/сущности, рёбра — отношения (совместная встречаемость,
риторический контраст, эмоциональная окраска).

Отвечай на основе контекста из графа. Указывай, какие именно концепты и связи
поддерживают твой ответ. Если данных в графе недостаточно — скажи об этом явно."""

QUERY_TEMPLATE = """Контекст из дискурс-графа:
{context}

---
Вопрос: {query}

Ответь развёрнуто, опираясь на концепты и связи из контекста выше."""


class DiscourseQA:
    """Q&A по дискурс-графу с выбором стратегии ретривала."""

    def __init__(
        self,
        G: nx.Graph,
        llm_client,  # OpenAI-совместимый клиент
        model: str = "deepseek/deepseek-chat",
        strategy: str = "lightrag",
        docs: list[dict] | None = None,
    ):
        self.G = G
        self.client = llm_client
        self.model = model
        self.strategy = strategy
        self.docs = docs  # корпус для сниппетов в walk-контексте

    def _retrieve(self, query: str) -> str:
        if self.strategy == "walk":
            return retrieve_walk(self.G, query, docs=self.docs)
        if self.strategy == "community":
            return retrieve_community(self.G, query)
        return retrieve_lightrag(self.G, query, docs=self.docs)

    def answer(self, query: str) -> str:
        context = self._retrieve(query)
        user_msg = QUERY_TEMPLATE.format(context=context, query=query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def answer_all(
        self, questions: list[str], verbose: bool = False
    ) -> dict[str, str]:
        """Ответить на список вопросов. Возвращает {вопрос: ответ}."""
        results = {}
        for q in questions:
            if verbose:
                print(f"[QA] Вопрос: {q}")
            results[q] = self.answer(q)
        return results


def flat_context_answer(
    docs: list[dict],
    query: str,
    llm_client,
    model: str = "deepseek/deepseek-chat",
    max_chars: int = 12000,
) -> str:
    """
    Baseline: длинный контекст без графа — весь корпус в одном промпте.
    Аналог подхода Gemini / DeepSeek long-context.
    """
    corpus = "\n\n".join(
        f"[{d.get('title', '')}]\n{d.get('text', '')}"
        for d in docs
    )[:max_chars]

    system = "Ты — аналитик текстов. Отвечай на вопросы на основе приведённого корпуса."
    user = f"Корпус текстов:\n{corpus}\n\n---\nВопрос: {query}"

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

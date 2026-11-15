"""Wiring the nodes into a graph, and running it.   >>> YOUR TURN <<<

Tests that define "done": tests/agent/test_graph.py

    START -> extract --(invoice?)--> audit -> decide --(escalate?)--> human_review -> record -> END
                 \\________(no invoice)_______/      \\_____________(otherwise)_____________/

Two kinds of edges: fixed ones (``add_edge``: always go there) and conditional ones
(``add_conditional_edges``: a routing function reads the state and returns the next node's name).
The routing functions are plain functions, so they are tested on their own.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from docaudit.agent.nodes import AuditNodes
from docaudit.agent.schemas import AgentOutcome, HumanDecision
from docaudit.agent.state import AgentState

AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def route_after_extract(state: AgentState) -> str:
    """``"audit"`` if there is an invoice, else ``"decide"`` (which will escalate the error)."""
    if state["invoice"]:
        return "audit"
    else:
        return "decide"


def route_after_decide(state: AgentState) -> str:
    """``"human_review"`` if the decision is ``escalate``, else ``"record"``."""
    if state["decision"].action == "escalate":
        return "human_review"
    else:
        return "record"


def build_graph(nodes: AuditNodes, checkpointer: BaseCheckpointSaver[Any]) -> AgentGraph:
    """Build and compile the graph in the diagram above.

    Requirements:
      * A ``StateGraph(AgentState)`` with five nodes named ``extract``, ``audit``, ``decide``,
        ``human_review`` and ``record``, using the methods of ``nodes``.
      * The fixed and conditional edges in the diagram. Pass the list of possible destinations
        to ``add_conditional_edges`` so the graph can be validated (and drawn) before it runs.
      * ``compile(checkpointer=checkpointer)``: without one, ``interrupt`` can't work.
    """
    builder = StateGraph(AgentState)
    node_list = [
        ("extract", nodes.extract),
        ("audit", nodes.audit),
        ("decide", nodes.decide),
        ("human_review", nodes.human_review),
        ("record", nodes.record),
    ]
    for name, func in node_list:
        builder.add_node(name, func)

    builder.add_edge(START, "extract")
    builder.add_conditional_edges("extract", route_after_extract, ["audit", "decide"])
    builder.add_edge("audit", "decide")
    builder.add_conditional_edges("decide", route_after_decide, ["human_review", "record"])
    builder.add_edge("human_review", "record")
    builder.add_edge("record", END)

    graph = builder.compile(checkpointer)
    return graph


def thread_config(document_id: str) -> RunnableConfig:
    """One checkpoint thread per document (provided)."""
    return {"configurable": {"thread_id": document_id}}


async def run_document(app: AgentGraph, document_id: str, document: str) -> AgentOutcome:
    """Run a document from the start, on the thread ``document_id``.

    Requirements:
      * ``ainvoke`` the graph with ``document_id``, ``document`` and an empty ``trail``, and
        ``thread_config(document_id)``.
      * If the result has the key ``"__interrupt__"``, the graph is waiting for a human: return
        ``status="waiting_for_human"`` with the decision. Otherwise ``status="done"`` with the
        decision and the record.
    """

    result = await app.ainvoke(
        {"document_id": document_id, "document": document, "trail": []}, thread_config(document_id)
    )

    if "__interrupt__" in result:
        status, record = "waiting_for_human", None
    else:
        status, record = "done", result["record"]
    decision = result["decision"]

    outcome = AgentOutcome(document_id=document_id, status=status, decision=decision, record=record)

    return outcome


async def resume_document(app: AgentGraph, document_id: str, human: HumanDecision) -> AgentOutcome:
    """Give a waiting document its human decision and let the graph finish.

    Requirements:
      * First check that the thread is actually waiting: ``(await app.aget_state(config))``
        has non-empty ``interrupts``. If not, raise ``ValueError``: resuming a thread that isn't
        paused would silently start it over.
      * ``ainvoke(Command(resume=human.model_dump()), config)`` and return the outcome, exactly
        as ``run_document`` does.
    """
    state = await app.aget_state(config=thread_config(document_id))
    if len(state.interrupts) == 0:
        raise ValueError(f"Document {document_id} doesn't have any interrupted thread")
    result = await app.ainvoke(
        Command(resume=human.model_dump()), config=thread_config(document_id)
    )

    status, decision, record = "done", result["decision"], result["record"]

    outcome = AgentOutcome(document_id=document_id, status=status, decision=decision, record=record)

    return outcome

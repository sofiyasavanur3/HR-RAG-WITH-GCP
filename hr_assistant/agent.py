"""16 · agent — the LLM + search tool + system prompt + memory, tied
together by LangChain's create_agent.

Two variants, differing only in which prompt (from prompts.py, 02) they run:
  create_hr_agent           — plain SYSTEM_PROMPT (red-team baseline)
  create_reliability_agent  — RELIABILITY_SYSTEM_PROMPT (every real path)
"""

from langchain.agents import create_agent
from hr_assistant import prompts


def _build_agent(llm, tools, system_prompt: str):
    """Shared by both variants. The checkpointer (InMemorySaver) was removed
    as LangGraph API now handles persistence automatically. This simplifies
    the setup while maintaining functionality."""
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


def create_hr_agent(llm, tools):
    """Plain agent — used only by the red-team plain baseline
    (pipeline.build_plain_assistant)."""
    return _build_agent(llm, tools, prompts.SYSTEM_PROMPT)


def create_reliability_agent(llm, tools):
    """The default agent for every real code path — reinforced scope prompt
    with identity lock and NOT_FOUND handling."""
    return _build_agent(llm, tools, prompts.RELIABILITY_SYSTEM_PROMPT)
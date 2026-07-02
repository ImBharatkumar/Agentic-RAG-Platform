import operator
from typing import TypedDict, Annotated, Any, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    original_query: str
    rewritten_query: str
    context: List[Any]
    answer: Any
    reflection: str
    iterations: int
    messages: Annotated[List[BaseMessage], add_messages]  # episodic memory
    route: str


import operator
from typing import TypedDict, Annotated, Any, List

class AgentState(TypedDict):
    original_query: str
    rewritten_query: str
    context: Annotated[List[Any], operator.add]
    answer: Any
    reflection: str
    iterations: int

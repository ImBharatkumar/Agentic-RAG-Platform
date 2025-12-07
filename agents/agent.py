from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from processing.rag_chain import rag_chain

# Example: Adding instructions
rag_agent = Agent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions correctly",
    instruction="""You are an agent that provides the relevant information or answers for user query.
When a user asks for the information:
1. use rag_chain tool to fextch information from the vector db.
2. If u dont get correct or relevant anser from the tool_call, convience user to ask another question
""",
tools=[FunctionTool(rag_chain)]
)

root_agent = rag_agent

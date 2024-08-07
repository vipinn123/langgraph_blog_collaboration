import getpass
import os
import env_util 
from time import sleep
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import END, StateGraph, START



#sets the env variables for the various keys
env_util.set_keys()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"


from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)



def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
#                " If you are unable to fully answer, that's OK, another assistant with different tools "
#                " will help where you left off. Execute what you can to make progress."
#                " If you or any of the other assistants have the final answer or deliverable,"
#                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)


from typing import Annotated
from langchain_google_vertexai import ChatVertexAI

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]
tool_node = ToolNode(tools)



import operator
from typing import Annotated, Sequence, TypedDict

from langchain_google_vertexai import ChatVertexAI


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

import functools

from langchain_core.messages import AIMessage


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }


"""Initiate the LLM on Vertex AI"""
llm = ChatVertexAI(
    model="gemini-1.5-pro-001",
    temperature=1,
    max_tokens=8192,
    max_retries=6,
    stop=None,
)

# Research agent and node
research_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You are an expert blogger. Your task is to create a medium blog article about the requested topic. You can use the tools at your disposal to perform the task.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# reviewer
reviewer_agent = create_agent(
    llm,
    [tavily_tool],
    system_message="You are a content reviewer. You will critically examine the content for factuality, readability and interpretability. You will respond with suggestions to REVISE if you do not like a response and provide detailed review comments on what you did not like in the content. You will respond with FINAL ANSWER if you like the response and provide detailed review comments on what you liked in the content.",
)
article_node = functools.partial(agent_node, agent=reviewer_agent, name="reviewer_generator")

# Either agent can decide to end
from typing import Literal


def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    sleep(3)
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("Researcher", research_node)
workflow.add_node("reviewer_generator", article_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {"continue": "reviewer_generator", "call_tool": "call_tool", "__end__": END},
)
workflow.add_conditional_edges(
    "reviewer_generator",
    router,
    {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
    #{"continue": "Researcher", "__end__": END},
)

workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",
        "reviewer_generator": "reviewer_generator",
    },
)
workflow.add_edge(START, "Researcher")
graph = workflow.compile()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import traceback


try:
    grph = graph.get_graph(xray=True)
    grph.print_ascii()


except Exception as e:
    # This requires some extra dependencies and is optional
    print(traceback.format_exc())
    pass


events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="I need to write an Introductory medium article about Generative AI Agentic frameworks like Langgraph, Autogen and crewai."
                "The article should start with an overview of Agents."
                "The article then needs provide a detailed introduction of the Langgraph, Autogen and crewai. The article should also provide the pro and cons for each of the 3 frameworks. "
                "Provide an overall conclusion section at the end of the article summarizing your understanding across the Agentic frameworks"
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    print(s)
    print("----")
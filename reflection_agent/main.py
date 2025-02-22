## Reflection Agent ##



from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph 
# END is a constant for LangGraph's default 'end' node. 
# When a node with this key is reached, LangGraph will stop execution.

from chains import generate_chain, reflect_chain
# Import from chains.py. These are the chains which will run in each node of our LangGraph graph.


# Constants
REFLECT = "reflect"
GENERATE = "generate"
# These are the keys for the nodes in our LangGraph graph.


# Nodes
def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})
# The 'generation_node' function will receive as an input the state.
# The 'state' is the list of messages that have been passed through the graph so far.
# The function will then run the 'generate_chain' chain, which will generate a response based on the state.

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]
# The 'reflection_node' function will also receive the sequence of messages.
# It will then invoke the 'reflect_chain' but instead will respond with a list of messages in the format of a HumanMessage.



# Initialize Graph
builder = MessageGraph()

# Define the nodes in the graph
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

# Tell the graph which node to start from (entry point)
builder.set_entry_point(GENERATE)


# Conditional branch
def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT
# Receives the state and returns the next node to go to based on the state.
# If the state has more than 6 messages, the graph will stop execution.
# Otherwise, it will continue to the REFLECT node.


builder.add_conditional_edges(GENERATE, should_continue)
# Add the conditional branch to the graph.
# This will determine which node to go to next based on the state.
builder.add_edge(REFLECT, GENERATE)
# Add an edge from the REFLECT node back to the GENERATE node.
# This will create a loop in the graph, allowing the conversation to continue indefinitely.


graph = builder.compile()
# Compile the graph to create the final structure.
# The graph is now ready to be executed
print(graph.get_graph().draw_mermaid())
# Print the graph in Mermaid format. Mermaid is OS tool for visualizing graphs.
# This will show the structure of the graph in a visual way.
# You can copy and paste this into a Mermaid editor to visualize the graph.
# https://mermaid.live/edit
graph.get_graph().print_ascii()
# Print the graph in ASCII format.
# This will show the structure of the graph in a text-based way.


if __name__ == "__main__":
    print("Hello LangGraph!")
    inputs = HumanMessage(content="""Make this tweet better:"
                          @LangChainAI
            - newly Tool Calling feature is seriously underrated.

            After a long wait, it's here- making the implementation of agents across different models with funtion calling.
            
            Made a video covering their newest blog post
                          """)
    response = graph.invoke(inputs)
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

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


# Graph
builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)





if __name__ == "__main__":
    print("Hello LangGraph!")
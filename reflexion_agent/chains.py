## Reflexion Agent ##

import datetime

from dotenv import load_dotenv

load_dotenv()


from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
# Handle the output of the OpenAI API
# Either parse the output as JSON or as Pydantic objects


from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion

llm = ChatOpenAI(model="gpt-4-turbo-preview")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])
# Define the OpenAI model
# Parse the output as JSON
# Parse the output as Pydantic objects using the AnswerQuestion schema


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher."
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
# Define the prompt template for the actor
# The MessagesPlaceholder will pass in all the message history to the prompt
# The partial function will fill in the current time

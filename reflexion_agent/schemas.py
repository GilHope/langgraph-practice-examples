from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")
# Define the reflection schema
# The reflection schema will be used to critique the initial answer
# The reflection schema has two fields: missing and superfluous


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(descriptio="Your reflection on the intial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
# Define the AnswerQuestion schema
# The AnswerQuestion schema will be used to answer the question
# The AnswerQuestion schema has two fields: answer and reflection
# The search_queries field is a list of search queries to improve the answer
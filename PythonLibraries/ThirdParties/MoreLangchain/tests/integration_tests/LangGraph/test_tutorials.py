from typing import TypedDict, List

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
class CountryInfo(BaseModel):
    capital: str = Field(description="Capital of the country")
    population: str = Field(description="Population of the country")
    
parser = PydanticOutputParser(pydantic_object=CountryInfo)

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(
    template="""
Give capital and population of India.

Return ONLY valid JSON.
Do NOT explain.
Do NOT add text.
Do NOT add markdown.

{format_instructions}
""",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

result = chain.invoke({})

print(result)
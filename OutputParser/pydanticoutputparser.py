from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define Output Schema
class CountryInfo(BaseModel):
    capital: str = Field(description="Capital of the country")
    population: str = Field(description="Population of the country")

# Step 2: Create Parser
parser = PydanticOutputParser(pydantic_object=CountryInfo)

# Step 3: LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Step 4: Prompt Template
prompt = PromptTemplate(
    template="Give capital and population of India.\n{format_instructions}",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Step 5: Chain
chain = prompt | llm | parser

# Step 6: Invoke
result = chain.invoke({})

print(result)
print(type(result))
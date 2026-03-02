from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


llm = ChatGroq(model="llama-3.1-8b-instant")

# Step 1: Define output schema
schemas = [
    ResponseSchema(name="capital", description="Capital of the country"),
    ResponseSchema(name="population", description="Population of the country")
]

# Step 2: Create output parser
parser = StructuredOutputParser.from_response_schemas(schemas)

# Step 3: Get formatting instructions
format_instructions = parser.get_format_instructions()

# Step 4: Create prompt
prompt = PromptTemplate(
    template="Give me the capital and population of India.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": format_instructions},
)

# Step 5: Chain
chain = prompt | llm | parser

# Step 6: Invoke
result = chain.invoke({})

print(result)
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Load document
loader = TextLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/XVnuuEg94sAE4S_xAsGxBA.txt")
docs = loader.load()

# Extract text
text = docs[0].page_content[:2000]   # limit size (VERY IMPORTANT)

# Parser
parser = JsonOutputParser()

# Strong prompt
prompt = PromptTemplate(
    template="""
You are an AI assistant.

Task:
1. Summarize the document.
2. Return a short exact excerpt from the document as source.

Rules:
- Return ONLY JSON
- No explanation
- No extra text

{format_instructions}

Document:
{text}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Chain
chain = prompt | llm | parser

result = chain.invoke({"text": text})

print(result)
from langchain_groq import ChatGroq
import os
from typing import TypedDict , Annotated
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.3-70b-versatile")
#schema 
#simpole TypeDict
class Review(TypedDict):
  summary : str
  sentiment : str

strcuture_model = model.with_structured_output(Review)

result = strcuture_model.invoke("AI review covers the analysis and evaluation of artificial intelligence models, tools, and their application across various fields. As of February 2026, top chatbots include Gemini, ChatGPT, and Copilot (4.0 rating), alongside specialized AI for code review, research, and customer feedback analysis.")

print(result)



#annoted type dict
class Review2(TypedDict):
  summary : Annotated[str , "brief summary of review"]
  sentiment : str

annoted_outpput = model.with_structured_output(Review2)  

reuslt2 = annoted_outpput.invoke("AI review covers the analysis and evaluation of artificial intelligence models, tools, and their application across various fields. As of February 2026, top chatbots include Gemini, ChatGPT, and Copilot (4.0 rating), alongside specialized AI for code review, research, and customer feedback analysis.")
print(reuslt2)
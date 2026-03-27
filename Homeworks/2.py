from langchain_groq import ChatGroq

import warnings
warnings.filterwarnings("ignore")


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
load_dotenv()


print("__________Exercise 1__________")
model1 = ChatGroq(model="llama-3.1-8b-instant" , temperature=0.8) #higher temp. for more creative answer 

model2 = ChatGroq(model="llama-3.1-8b-instant" , temperature=0.1) #lower tem. for more deterministic responses

prompts = ["Write a short poem about artificial intelligence.",
           "What are the key components of a neuralnetwork?",
           "List 5 tips for effective time management."]

print("model 1 : temp -> 0.8")
for prommpt in prompts:
  result = model1.invoke(prommpt)
  print(result.content)
  

print("model 2 : temp -> 0.1") 
for prompt in prompts:
  result = model2.invoke(prompts)
  print(result.content) 




#exercise 2
print("_______________Exercise 2___________________")

#created json parser
json_parse = JsonOutputParser()

format_instruction = """
RESPONSE FORMAT: Return ONLY a single JSON object-no markdown, no examples, no extra keys.
{
"title": "movie title",
"director": "director name",
"year": 2000,
"genre": "movie genre"
}
IMPORTANT: Your response must be *only* that JSON. DO NOT include any illustrative or example JSON."*
"""

prompt_template = PromptTemplate(
  template="""" You are a JSON-only assistant.
Task: Generate info about the movie "{movie_name}" in JSON format.
{format_instruction}""",
input_variables=["movie_name"],
partial_variables={"format_instruction" : format_instruction},
)

movie_chain = prompt_template | model2 | json_parse

movie_name = "The aveangers"

result = movie_chain.invoke(movie_name)
print("Parsed result:")
print(f"Title: {result['title']}")
print(f"Director: {result['director']}")
print(f"Year: {result['year']}")
print(f"Genre: {result['genre']}")
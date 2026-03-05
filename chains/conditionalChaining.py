from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.1-8b-instant")

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the following feedback as positive or negative\n{feedback}\n{format_instruction}",
    input_variables=["feedback"],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

positive_prompt = PromptTemplate(
    template="Write a thank you response for this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="Write an apology and improvement response for this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

parser = StrOutputParser()

branch = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'positive', positive_prompt | model | parser),
    (lambda x: x['sentiment'].sentiment == 'negative', negative_prompt | model | parser),
    negative_prompt | model | parser  # default
)

full_chain = RunnableLambda(lambda x: {
    'sentiment': classifier_chain.invoke({'feedback': x['feedback']}),
    'feedback': x['feedback']
}) | branch

result = full_chain.invoke({'feedback': "This product is amazing! I love it!"})
print(result)
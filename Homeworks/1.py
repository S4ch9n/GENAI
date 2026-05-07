import warnings
warnings.filterwarnings("ignore")

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

llm = ChatGroq(model="llama-3.1-8b-instant")

# basic pronpt
print("basic prommpt")
prompts = [
  "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The benefits of sustainable energy include"
]
for prompt in prompts:
  result = llm.invoke(prompt)
  print(result.content)
  

# zero prompts
print("zero shot prompts")
zero_prompts = [
  # "Classify the following movie review as positive or negative: 'The movie I watched was awesome!'",
  # "Summarize the following paragraph , Climate change refers to long-term shifts in temperatures and weather patterns, primarily driven by human activities like burning fossil fuels, deforestation, and industrial processes since the 1800s.",
  "Translate the following English phrase to Spanish: With that sharingan how far you can see ?"
]
for prompt in zero_prompts:
  result = llm.invoke(prompt)
  print(result.content)



# one shot prompts
print("One shot prompts")
one_shot = [
  # Create a prompt with one example of a formal email, then ask the model to write another formal email on a different topic.
  """
  This is an example of formal email
  Subject: Leave Request

Dear Sir,

I am writing to request a leave of absence 
for two days due to a family emergency.

Regards,
Sachin 
Now write another formal email on a different topic""",

# Provide one example of converting a technical concept into a simple explanation, then ask the model to explain a different concept.
"""Techincal Concept : Computer understand machine language
Simple Explaination : Computer convert texts to binary digits like 00101 with the help of compiler and interpreter and understand it
Technical Concept : An API is an application program interface
Simple Explaination : """,

#Give one example of extracting keywords from a sentence, then ask the model to extract keywords from a new sentence.
'''Sentence: "I love eating pizza in New York"
Keywords: love, eating, pizza, New York
Sentence : "I like to live alone , as iam a introvert"
Keywords : '''
]
for prompt in one_shot:
  result = llm.invoke(prompt)
  print(result.content)




print("few shot prompts")
few_shot = [
  """Examples of classifying emotions in statements:
Statement: 'I just won my first marathon!'
Emotion: Joy
Statement: 'I can't believe I lost my keys again.'
Emotion: Frustration
Statement: 'My best friend is moving to another country.'
Emotion: Sadness
Now, classify the emotion in the following statement:
Statement: 'That movie was so scary I had to cover my eyes.'
Emotion: """
]
for prompt in few_shot:
  result = llm.invoke(prompt)
  print(result.content)




#chain of thought
print("chain of thought")
cot = [
  #Write a prompt that asks the model to think through whether a student should study tonight or go to a movie with friends, considering their upcoming test in two days.
  """Consider the situation : "A student has a test in two days. 
  He cover 50% of the syllabus
His friends are inviting him to a movie tonight.

Think step by step considering : 
1. How much time does the student have?
2. How prepared is the student?
3. What are consequences of each choice?
4. Is balance possible?
 """,
 """Consider the situation: Someone wants to make a 
peanut butter and jelly sandwich for the first time.

Think step by step considering:
1. What ingredients are needed?
2. What tools are needed?
3. What is the correct order of steps?
4. Any tips for making the perfect sandwich?

Explain the complete process step by step."""
]
for prompt in cot:
  result = llm.invoke(prompt)
  print(result.content)
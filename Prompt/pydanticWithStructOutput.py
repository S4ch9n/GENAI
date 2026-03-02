from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.3-70b-versatile")

# Pydantic Schema
class Review(BaseModel):
    name: str = Field(description="name of the product being reviewed")
    summary: str = Field(description="brief summary of the review")
    sentiment: str = Field(description="overall sentiment of the review")
    pros: list[str] = Field(description="list of positive points")
    cons: list[str] = Field(description="list of negative points")
    rating: float = Field(description="rating out of 10")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
iPhone 15 Pro Review:
The iPhone 15 Pro is an outstanding smartphone with a titanium build 
that feels premium. The A17 Pro chip is blazing fast and the camera 
system is exceptional. However the battery life could be better and 
it is quite expensive. Overall a great phone for professionals.
""")

print(result)
print("Name:", result.name)
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Rating:", result.rating)
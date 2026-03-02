from langchain_groq import ChatGroq
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

model = ChatGroq(model="llama-3.3-70b-versatile")

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "name of product"},
        "description": {"type": "string", "description": "brief description"},
        "sentiment": {"type": "string", "description": "positive or negative"},
        "pros": {"type": "array", "items": {"type": "string"}, "description": "list of pros"},
        "cons": {"type": "array", "items": {"type": "string"}, "description": "list of cons"},
        "rating": {"type": "number", "description": "rating out of 10"}
    },
    "required": ["title", "description", "sentiment", "pros", "cons", "rating"]
}
structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
iPhone 15 Pro Review:
The iPhone 15 Pro is an outstanding smartphone with a titanium build 
that feels premium. The A17 Pro chip is blazing fast and the camera 
system is exceptional. However the battery life could be better and 
it is quite expensive. Overall a great phone for professionals.
""")

print(result)
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import warnings
warnings.filterwarnings("ignore")

llm = HuggingFacePipeline.from_model_id(
  model_id = "google/flan-t5-small",
  task = "text2text-generation",
  pipeline_kwargs = dict(
    max_new_tokens = 100,
  )
)

result = llm.invoke("What is the capital of India ? ")
print(result.content)
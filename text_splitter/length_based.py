from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=100,    # each chunk max 100 characters
    chunk_overlap=20,  # 20 characters overlap between chunks
    separator = "\n"
)
text = """
Generative AI (GenAI) is a transformative branch of artificial intelligence that creates new, original content—including text, images, code, and music—by identifying patterns in vast datasets. Unlike traditional AI, which analyzes existing data, GenAI uses models like transformers to produce novel outputs. It boosts productivity and fosters innovation across industries, but raises ethical risks regarding bias and job displacement.
Key Aspects of GenAI:
Core Function: Instead of simply classifying or identifying data, GenAI models like ChatGPT (text) or DALL-E (images) generate, synthesize, and create content that mimics human creativity.
Technological Basis: Built on machine learning techniques, particularly neural networks like Generative Adversarial Networks (GANs) and Transformers, these systems learn from massive datasets to produce high-quality, relevant outputs.
Applications: The technology is transforming sectors by accelerating content creation, automating tedious coding tasks, simulating healthcare scenarios, and designing new products.
Ethical Challenges: The rise of GenAI brings significant concerns, including the creation of deepfakes, the spread of misinformation, potential biases in output, and the displacement of creative jobs. 
"""
result = splitter.split_text(text)
print(result)
print(len(result))  # number of chunks
print(result[0])    # first chunk
print(result[1])    # second chunk
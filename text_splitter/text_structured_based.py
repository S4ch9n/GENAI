from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,    # each chunk max 100 characters
    chunk_overlap=20,  # 20 characters overlap between chunks
)
text = """
Generative AI (GenAI) is a transformative branch of artificial intelligence that creates new, original content—including text, images, code, and music—by identifying patterns in vast datasets. Unlike traditional AI, which analyzes existing data, GenAI uses models like transformers to produce novel outputs. It boosts productivity and fosters innovation across industries, but raises ethical risks regarding bias and job displacement.
Key Aspects of GenAI:
Core Function: Instead of simply classifying or identifying data, GenAI models like ChatGPT (text) or DALL-E (images) generate, synthesize, and create content that mimics human creativity.
Technological Basis: Built on machine learning techniques, particularly neural networks like Generative Adversarial Networks (GANs) and Transformers, these systems learn from massive datasets to produce high-quality, relevant outputs.
Applications: The technology is transforming sectors by accelerating content creation, automating tedious coding tasks, simulating healthcare scenarios, and designing new products.
Ethical Challenges: The rise of GenAI brings significant concerns, including the creation of deepfakes, the spread of misinformation, potential biases in output, and the displacement of creative jobs. 
"""
chunks = splitter.split_text(text)
print(len(chunks))  # number of chunks
print(chunks)
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# Introduction to AI
Artificial intelligence is transforming the world.

## Machine Learning
Machine learning is a subset of AI that learns from data.

### Supervised Learning
Supervised learning uses labeled data to train models.

### Unsupervised Learning
Unsupervised learning finds patterns in unlabeled data.

## Deep Learning
Deep learning uses neural networks with many layers.

### CNN
CNNs are used for image recognition tasks.

### RNN
RNNs are used for sequential data like text.
"""

headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

chunks = splitter.split_text(markdown_text)

for chunk in chunks:
    print(chunk)
    print("---")
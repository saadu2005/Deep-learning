"""File purpose: Explain how artificial intelligence, machine learning, and deep learning relate to one another.

Explanation: The script stores a short definition for each term and prints the definitions in a readable form.

Real-life example: A photo-organizing app is an AI product; it may use machine learning to learn from examples and deep learning to recognize objects in photos.
"""

# ==========================================
# AI vs ML vs Deep Learning
# ==========================================

concepts = {
    "AI": "The broad field of creating systems that perform intelligent tasks.",
    "Machine Learning": "A part of AI where algorithms learn patterns from data.",
    "Deep Learning": "A part of Machine Learning that commonly uses multi-layer neural networks."
}

# ==========================================
# Display Concepts
# ==========================================
for name, description in concepts.items():
    print(f"\n{name}")
    print("-" * len(name))
    print(description)

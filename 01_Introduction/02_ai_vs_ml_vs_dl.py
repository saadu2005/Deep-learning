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

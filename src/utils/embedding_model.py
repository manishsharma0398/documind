"""The embedding model and the vector width it produces.

These two travel together on purpose. The collection is created with a fixed
vector size, so a dimension that disagrees with the model builds an index that
rejects every point. Anything importing one almost always needs the other.
"""

EMBEDDING_MODEL = "text-embedding-3-small"

# Must match what EMBEDDING_MODEL produces.
EMBEDDING_DIMENSIONS: int = 1536

# USD per million tokens, for the ingest cost line. Model-specific: changing
# EMBEDDING_MODEL without changing this silently misreports every figure.
EMBEDDING_COST_PER_MILLION = 0.02

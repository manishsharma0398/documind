# Bytes inspected when deciding whether a file is binary.
SNIFF_BYTES = 4096

# Guardrails so one request cannot sweep up an entire filesystem.
MAX_FILES = 2_000
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 50_000_000


SKIP_DIRS = {
    ".venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".next",
    "dist",
    "build",
    "target",
    "coverage",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "qdrant_storage",
    ".ssh",
    ".aws",
    ".gnupg",
    ".kube",
}

# Never ingested, whatever extensions the caller asks for.
DENY_NAMES = {
    ".env",
    ".npmrc",
    ".netrc",
    ".pgpass",
    ".htpasswd",
    ".git-credentials",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
DENY_SUFFIXES = {
    ".pem",
    ".key",
    ".pfx",
    ".p12",
    ".jks",
    ".keystore",
    ".ppk",
    ".gpg",
    ".asc",
    ".kdbx",
}


# Generated lock files: pure noise, never worth embedding.
SKIP_NAMES = {
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pnpm-lock.yaml",
    "bun.lockb",
    ".terraform.lock.hcl",
}

SKIP_SUFFIXES = {".lock"}

# 100 left too little room: the section breadcrumb was ~29% of every embedded
# chunk, and top_k=3 gave the model ~300 tokens to answer from. At 400 the
# breadcrumb is ~17% and top_k=3 is ~510 tokens.
#
# Chunks come out at ~178 tokens on average, well under this number: the
# markdown header split runs first, and most sections are shorter than the
# budget. This is a ceiling, not a target.
TOKEN_SIZE = 400
TEXT_OVERLAP = int(TOKEN_SIZE * 0.1)

EMBEDDING_MODEL = "text-embedding-3-small"
MIN_CHUNK_TOKENS = 100

# Must match the embedding model that produces the vectors.
EMBEDDING_DIMENSIONS: int = 1536
DEFAULT_TOP_K: int = 3

OPENAI_TIMEOUT: float = 500
OPENAI_MAX_RETRIES: int = 5

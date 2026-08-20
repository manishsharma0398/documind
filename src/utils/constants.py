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

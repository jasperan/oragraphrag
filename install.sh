#!/usr/bin/env bash
# OraGraphRAG one-shot setup script.
#
# This sets up the conda env, installs the package, optionally brings up the
# Oracle 23ai Free container, and applies the two operator fixes documented
# in the plan (vector_memory_size + USERS tablespace) so init-db succeeds.

set -euo pipefail

ENV_NAME="oragraphrag"
PYTHON_VERSION="3.12"
ORACLE_PASSWORD="${ORACLE_PASSWORD:-Welcome12345*}"

color() { printf '\033[%sm%s\033[0m\n' "$1" "$2"; }
info()  { color "1;34" "[install] $*"; }
warn()  { color "1;33" "[install] $*"; }
fail()  { color "1;31" "[install] $*"; exit 1; }

# 1. conda env
if ! command -v conda >/dev/null 2>&1; then
  fail "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "creating conda env $ENV_NAME (python $PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
else
  info "conda env $ENV_NAME already exists"
fi

# 2. pip install in editable mode with notebook extras
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

info "installing oragraphrag in editable mode (+notebook extras)"
pip install -e ".[notebook]" --quiet

# Dev deps live in the PEP 735 dependency-groups table; older pip falls back
# to direct install.
if pip install --group dev . --dry-run >/dev/null 2>&1; then
  pip install --group dev . --quiet
else
  warn "pip <25.1 detected; installing dev deps directly"
  pip install pytest pytest-asyncio pytest-cov ruff black mypy types-PyYAML --quiet
fi

# 3. config + .env from templates
if [ ! -f config.yaml ]; then
  cp config.yaml.example config.yaml
  info "wrote config.yaml from template (edit before init-db)"
fi
if [ ! -f .env ]; then
  cp .env.example .env
  info "wrote .env from template"
fi

# 4. Optional: bring up the Oracle container
if [ "${ORAGRAPHRAG_SETUP_ORACLE:-0}" = "1" ]; then
  info "starting Oracle 23ai Free container"
  docker compose up -d oracle-free

  info "waiting for Oracle to be healthy (this takes ~60-90s)"
  for i in $(seq 1 30); do
    if docker exec oragraphrag-oracle /opt/oracle/checkDBStatus.sh >/dev/null 2>&1; then
      info "Oracle is healthy after ${i}0s"
      break
    fi
    sleep 10
    if [ "$i" = "30" ]; then
      fail "Oracle did not become healthy after 5 minutes"
    fi
  done

  info "applying operator-setup fixes (vector_memory_size, USERS tablespace, ORAGRAPH user)"
  docker exec -i oragraphrag-oracle sqlplus -S "system/${ORACLE_PASSWORD}@//localhost:1521/FREEPDB1" <<SQL || true
-- Enable VECTOR memory pool (required for HNSW indexes).
ALTER SYSTEM SET vector_memory_size=512M SCOPE=SPFILE;
EXIT;
SQL

  info "restarting Oracle to apply vector_memory_size"
  docker exec -i oragraphrag-oracle sqlplus -S "/ as sysdba" <<SQL || true
SHUTDOWN IMMEDIATE;
STARTUP;
EXIT;
SQL

  # Wait for DB to come back up.
  for i in $(seq 1 30); do
    if docker exec oragraphrag-oracle /opt/oracle/checkDBStatus.sh >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done

  info "creating USERS tablespace + ORAGRAPH user with grants"
  docker exec -i oragraphrag-oracle sqlplus -S "system/${ORACLE_PASSWORD}@//localhost:1521/FREEPDB1" <<SQL || true
CREATE TABLESPACE USERS
    DATAFILE 'users01.dbf' SIZE 200M AUTOEXTEND ON NEXT 50M MAXSIZE 2G
    EXTENT MANAGEMENT LOCAL
    SEGMENT SPACE MANAGEMENT AUTO;
CREATE USER ORAGRAPH IDENTIFIED BY "${ORACLE_PASSWORD}";
GRANT CONNECT, RESOURCE TO ORAGRAPH;
GRANT UNLIMITED TABLESPACE TO ORAGRAPH;
GRANT CREATE PROPERTY GRAPH TO ORAGRAPH;
ALTER USER ORAGRAPH DEFAULT TABLESPACE USERS;
ALTER USER ORAGRAPH QUOTA UNLIMITED ON USERS;
EXIT;
SQL
fi

# 5. Optional: Ollama models
if [ "${ORAGRAPHRAG_SETUP_OLLAMA:-0}" = "1" ]; then
  info "starting Ollama container"
  docker compose up -d ollama
  sleep 5
  info "pulling default models (gemma3:270m for chat, nomic-embed-text for embeddings)"
  docker exec oragraphrag-ollama ollama pull gemma3:270m
  docker exec oragraphrag-ollama ollama pull nomic-embed-text
fi

info "install complete"
info "next: conda activate $ENV_NAME && oragraphrag init-db --rebuild && oragraphrag graphify <folder>"

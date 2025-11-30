#!/bin/bash
# Script para lanzar la búsqueda de hiperparámetros con Optuna

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Búsqueda de hiperparámetros con Optuna${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ uv no encontrado${NC}"
    echo -e "${YELLOW}Instala uv con: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Entorno virtual no encontrado${NC}"
    echo -e "${GREEN}Creando entorno virtual y sincronizando dependencias...${NC}"
    uv sync --python 3.10
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Error sincronizando dependencias${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Ejecutando búsqueda con Optuna...${NC}"
uv run python envs/train_optuna_ppo.py "$@"


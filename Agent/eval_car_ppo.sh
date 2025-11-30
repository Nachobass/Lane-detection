#!/bin/bash
# Script para evaluar un modelo PPO entrenado en Unity

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Evaluación de Modelo PPO${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Verificar que uv está instalado
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ uv no encontrado${NC}"
    echo -e "${YELLOW}Instala uv con: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

# Cambiar al directorio del script
cd "$SCRIPT_DIR"

# Verificar que el venv existe, si no, sincronizar dependencias
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Entorno virtual no encontrado${NC}"
    echo -e "${GREEN}Creando entorno virtual y sincronizando dependencias...${NC}"
    uv sync --python 3.10
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ Error sincronizando dependencias${NC}"
        exit 1
    fi
fi

# Ejecutar el script de evaluación con uv
echo -e "${GREEN}Ejecutando evaluación...${NC}"
uv run python envs/eval_car_ppo.py "$@"


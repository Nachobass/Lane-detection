#!/bin/bash
# Script para visualizar los logs de TensorBoard

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Obtener el directorio de logs
TENSORBOARD_DIR="${1:-results/custom_ppo/tensorboard}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Iniciando TensorBoard${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Directorio de logs: ${YELLOW}$TENSORBOARD_DIR${NC}"
echo ""
echo -e "${GREEN}TensorBoard estará disponible en:${NC} http://localhost:6006"
echo -e "${YELLOW}Presiona Ctrl+C para detener TensorBoard${NC}"
echo ""

# Directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verificar que uv está instalado
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ uv no encontrado${NC}"
    echo -e "${YELLOW}Instala uv con: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    exit 1
fi

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

# Ejecutar TensorBoard
# Primero intentar arreglar el shebang del ejecutable si es necesario
if [ -f ".venv/bin/tensorboard" ]; then
    # Verificar y corregir el shebang si apunta a una ruta incorrecta
    CURRENT_PYTHON="$(pwd)/.venv/bin/python"
    SHEBANG_LINE=$(head -1 .venv/bin/tensorboard)
    if [[ "$SHEBANG_LINE" != "#!$CURRENT_PYTHON" ]] && [[ "$SHEBANG_LINE" == "#!"* ]]; then
        # El shebang apunta a una ruta diferente, corregirlo
        sed -i "1s|.*|#!$CURRENT_PYTHON|" .venv/bin/tensorboard 2>/dev/null || true
    fi
    # Intentar usar el ejecutable
    .venv/bin/tensorboard --logdir "$TENSORBOARD_DIR" --port 6006
else
    # Si no existe el ejecutable, usar python directamente
    .venv/bin/python -c "import sys; sys.argv = ['tensorboard', '--logdir', '$TENSORBOARD_DIR', '--port', '6006']; from tensorboard import main; main.run_main()"
fi


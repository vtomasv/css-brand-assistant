#!/bin/bash
# CSS Brand Assistant — Script de instalación manual
# Uso: bash setup.sh (desde el directorio del plugin)
# Este script es un fallback para cuando la instalación desde Pinokio falla.

set -e

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Directorio del plugin: $PLUGIN_DIR"
cd "$PLUGIN_DIR"

# Crear venv si no existe
if [ ! -d "venv" ]; then
  echo "🐍 Creando entorno virtual Python..."
  python3 -m venv venv
fi

# Instalar dependencias SIEMPRE con el pip del venv
echo "📦 Instalando dependencias Python en el venv..."
"$PLUGIN_DIR/venv/bin/pip" install --upgrade pip --quiet
"$PLUGIN_DIR/venv/bin/pip" install -r requirements.txt

# Verificar instalación
echo "✅ Verificando instalación..."
"$PLUGIN_DIR/venv/bin/python" -c "import fastapi, uvicorn, requests, bs4; print('OK: todas las dependencias instaladas correctamente')"

# Crear directorios de datos
echo "💾 Inicializando directorios de datos..."
mkdir -p data/agents data/prompts/system data/sessions data/exports data/brands data/campaigns data/audit

# Copiar defaults si no existen
[ -f "data/agents/agents.json" ] || cp defaults/agents.json data/agents/agents.json 2>/dev/null || true
cp defaults/prompts/*.md data/prompts/system/ 2>/dev/null || true

echo ""
echo "✅ Instalación completada. Ahora puedes iniciar el plugin desde Pinokio."

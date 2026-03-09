# pull_model.ps1 — Descarga el modelo LLM correcto según la RAM disponible
# Usa la ruta completa de Ollama para evitar problemas de PATH en conda

param(
    [string]$Model = ""
)

# Determinar la ruta de Ollama
$ollamaPath = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollamaPath) {
    $ollamaPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
}
if (-not (Test-Path $ollamaPath)) {
    # Intentar ruta alternativa de instalación de sistema
    $ollamaPath = "C:\Program Files\Ollama\ollama.exe"
}
if (-not (Test-Path $ollamaPath)) {
    Write-Host "ERROR: No se encontro Ollama en ninguna ruta conocida."
    Write-Host "Por favor instala Ollama desde https://ollama.com/download"
    exit 1
}

Write-Host "Ollama encontrado en: $ollamaPath"

# Si no se especificó modelo, elegir según RAM disponible
if (-not $Model) {
    $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    Write-Host "RAM detectada: ${ramGB}GB"
    if ($ramGB -lt 6) {
        $Model = "llama3.2:1b"
    } elseif ($ramGB -lt 12) {
        $Model = "llama3.2:3b"
    } else {
        $Model = "llama3.1:8b"
    }
}

Write-Host "Descargando modelo: $Model"
Write-Host "Esto puede tardar varios minutos dependiendo de tu conexion..."

# Verificar si el modelo ya está descargado
$listOutput = & $ollamaPath list 2>&1
if ($listOutput -match [regex]::Escape($Model.Split(":")[0])) {
    Write-Host "MODELO_YA_DISPONIBLE: $Model ya esta instalado"
    exit 0
}

# Descargar el modelo
& $ollamaPath pull $Model
if ($LASTEXITCODE -eq 0) {
    Write-Host "MODELO_DESCARGADO: $Model descargado correctamente"
} else {
    Write-Host "ERROR: Fallo al descargar $Model (codigo $LASTEXITCODE)"
    exit 1
}

# install_ollama.ps1 - Instala Ollama en Windows si no esta presente
# Ejecutar como: powershell -ExecutionPolicy Bypass -File scripts\install_ollama.ps1

$ErrorActionPreference = "Continue"

# Buscar ollama en el PATH
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue

if ($ollamaCmd) {
    Write-Host "OLLAMA_ALREADY_INSTALLED: $($ollamaCmd.Source)"
    exit 0
}

# Buscar en la ruta de instalacion por defecto de Windows
$defaultPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (Test-Path $defaultPath) {
    Write-Host "OLLAMA_ALREADY_INSTALLED: $defaultPath"
    exit 0
}

# Descargar e instalar Ollama silenciosamente
Write-Host "Descargando instalador de Ollama..."
$installer = Join-Path $env:TEMP "OllamaSetup.exe"

try {
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer -UseBasicParsing
    Write-Host "Instalando Ollama silenciosamente..."
    Start-Process -FilePath $installer -ArgumentList "/S" -Wait -NoNewWindow
    Write-Host "OLLAMA_INSTALLED"
} catch {
    Write-Host "ERROR: No se pudo instalar Ollama: $_"
    Write-Host "Por favor instala Ollama manualmente desde https://ollama.com/download"
    exit 1
}

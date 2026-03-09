# start_ollama.ps1 - Inicia Ollama en segundo plano en Windows
# Ejecutar como: powershell -ExecutionPolicy Bypass -File scripts\start_ollama.ps1

$ErrorActionPreference = "Continue"

# Verificar si Ollama ya esta corriendo
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "OLLAMA_ALREADY_RUNNING"
        exit 0
    }
} catch {
    # No esta corriendo, continuar
}

# Buscar el ejecutable de ollama
$ollamaPath = $null

$cmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($cmd) {
    $ollamaPath = $cmd.Source
}

if (-not $ollamaPath) {
    $defaultPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $defaultPath) {
        $ollamaPath = $defaultPath
    }
}

if (-not $ollamaPath) {
    Write-Host "OLLAMA_NOT_FOUND: Instala Ollama desde https://ollama.com/download"
    exit 1
}

# Iniciar Ollama en segundo plano sin ventana
Write-Host "Iniciando Ollama desde: $ollamaPath"
Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden -PassThru | Out-Null
Write-Host "OLLAMA_STARTED"

# Esperar a que este listo
Start-Sleep -Seconds 4
Write-Host "OLLAMA_READY"

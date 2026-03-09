# setup_venv.ps1 - Crea el entorno virtual e instala dependencias en Windows
# Usa el Python del venv directamente con -m pip para evitar interferencia de conda
# Ejecutar como: powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1

param(
    [string]$ProjectDir = $PSScriptRoot + "\.."
)

$ErrorActionPreference = "Stop"

# Resolver ruta absoluta del proyecto
$ProjectDir = Resolve-Path $ProjectDir
Write-Host "Directorio del proyecto: $ProjectDir"

# Ruta al venv
$venvDir = Join-Path $ProjectDir "venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirementsFile = Join-Path $ProjectDir "requirements.txt"

# Paso 1: Crear el venv si no existe o si python.exe no esta dentro
if (-not (Test-Path $venvPython)) {
    Write-Host "Creando entorno virtual Python..."
    # Usar el python del sistema (no el del venv)
    $systemPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $systemPython) {
        Write-Host "ERROR: Python no encontrado en el PATH"
        exit 1
    }
    Write-Host "Python del sistema: $systemPython"
    & $systemPython -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo crear el entorno virtual"
        exit 1
    }
    Write-Host "Entorno virtual creado en: $venvDir"
} else {
    Write-Host "Entorno virtual ya existe: $venvDir"
}

# Verificar que python.exe existe en el venv
if (-not (Test-Path $venvPython)) {
    Write-Host "ERROR: $venvPython no encontrado despues de crear el venv"
    exit 1
}

Write-Host "Python del venv: $venvPython"

# Paso 2: Actualizar pip usando el Python del venv
Write-Host "Actualizando pip..."
& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "ADVERTENCIA: No se pudo actualizar pip, continuando..."
}

# Paso 3: Instalar dependencias usando el Python del venv
Write-Host "Instalando dependencias desde requirements.txt..."
& $venvPython -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo la instalacion de dependencias"
    exit 1
}

# Paso 4: Verificar que los modulos criticos estan instalados
Write-Host "Verificando modulos criticos..."
$modules = @("fastapi", "uvicorn", "requests", "pydantic")
foreach ($mod in $modules) {
    $result = & $venvPython -c "import $mod; print('OK: $mod')" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Modulo '$mod' no instalado correctamente"
        exit 1
    }
    Write-Host $result
}

Write-Host ""
Write-Host "DEPS_OK: Todas las dependencias instaladas correctamente"

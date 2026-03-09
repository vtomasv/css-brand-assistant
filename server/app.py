"""
CSS Brand Assistant — Backend FastAPI
Plugin de Pinokio para gestión de ADN de marca y campañas digitales con IA local.

Arquitectura:
  - Módulo de Marcas: CRUD de marcas y onboarding
  - Módulo de ADN: construcción, versionado y refinamiento del ADN empresarial
  - Módulo de Campañas: creación, planificación temporal y publicaciones
  - Módulo de Agentes: orquestación de LLMs locales vía Ollama
  - Módulo de Auditoría: trazabilidad de todas las operaciones de agentes
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# Forzar UTF-8 en stdout/stderr para Windows (evita UnicodeEncodeError con tildes/ñ)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuración de rutas (siempre absolutas desde __file__)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()   # raíz del plugin
APP_DIR  = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
DEFAULTS_DIR = BASE_DIR / "defaults"

PORT = int(os.environ.get("PORT", 7860))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Timeouts para llamadas a Ollama
# Se pueden sobreescribir con variables de entorno o via config.json
# (clave: "ollama_timeout", "ollama_timeout_campaign", "ollama_timeout_adn")
# ---------------------------------------------------------------------------
# Timeout base para consultas cortas (ADN, regeneración de posts, etc.)
OLLAMA_TIMEOUT_DEFAULT: int = int(os.environ.get("OLLAMA_TIMEOUT", 300))       # 5 min
# Timeout para generación de campañas completas (muchas publicaciones)
OLLAMA_TIMEOUT_CAMPAIGN: int = int(os.environ.get("OLLAMA_TIMEOUT_CAMPAIGN", 600))  # 10 min
# Timeout para análisis de ADN de marca
OLLAMA_TIMEOUT_ADN: int = int(os.environ.get("OLLAMA_TIMEOUT_ADN", 300))       # 5 min

# ---------------------------------------------------------------------------
# Proveedores de generación de imágenes
# Se pueden configurar via variables de entorno o config.json
# Proveedores soportados: "ollama" | "automatic1111" | "comfyui" | "auto"
# - "auto": prueba en orden Ollama → A1111 → ComfyUI → placeholder SVG
# - "ollama": solo Ollama (macOS/Linux)
# - "automatic1111": AUTOMATIC1111 WebUI (http://localhost:7860/sdapi/v1/txt2img)
# - "comfyui": ComfyUI (http://localhost:8188)
# ---------------------------------------------------------------------------
IMAGE_PROVIDER: str = os.environ.get("IMAGE_PROVIDER", "auto")
A1111_URL: str = os.environ.get("A1111_URL", "http://localhost:7860")
COMFYUI_URL: str = os.environ.get("COMFYUI_URL", "http://localhost:8188")
IMAGE_TIMEOUT: int = int(os.environ.get("IMAGE_TIMEOUT", 300))  # 5 min

# ---------------------------------------------------------------------------
# Motor de imagen embebido (HuggingFace Diffusers + LCM)
# Permite generar imágenes localmente sin depender de Ollama, A1111 ni ComfyUI.
# El motor se carga en background al primer uso o al iniciar el servidor.
# ---------------------------------------------------------------------------
try:
    from image_engine import (
        generate_image as _engine_generate,
        load_engine_async as _engine_load_async,
        get_engine_status as _engine_get_status,
        is_engine_ready as _engine_is_ready,
        unload_engine as _engine_unload,
        list_available_models as _engine_list_models,
        DEFAULT_DIFFUSION_MODEL,
    )
    IMAGE_ENGINE_AVAILABLE = True
    logger_tmp = logging.getLogger("css-brand-assistant")
    logger_tmp.info("Motor de imagen embebido (Diffusers) disponible")
except ImportError as _ie:
    IMAGE_ENGINE_AVAILABLE = False
    DEFAULT_DIFFUSION_MODEL = "SimianLuo/LCM_Dreamshaper_v7"
    logging.getLogger("css-brand-assistant").warning(
        f"Motor de imagen embebido no disponible: {_ie}. "
        "Instala: pip install torch diffusers transformers accelerate safetensors"
    )

# ---------------------------------------------------------------------------
# Estado global de descargas de modelos en progreso
# Clave: nombre del modelo, Valor: dict con status/progress/error
# ---------------------------------------------------------------------------
_pull_status: Dict[str, Dict] = {}  # {model: {status, progress, error, started_at}}
_pull_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("css-brand-assistant")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CSS Brand Assistant",
    description="Plugin Pinokio para ADN de marca y campañas digitales con IA local",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: crear directorios y copiar defaults
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Inicializa directorios de datos y copia configuraciones por defecto."""
    for subdir in ["agents", "prompts/system", "sessions", "exports", "brands", "campaigns", "audit"]:
        (DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Copiar defaults de agentes si no existen
    defaults_agents = DEFAULTS_DIR / "agents.json"
    data_agents = DATA_DIR / "agents" / "agents.json"
    if defaults_agents.exists() and not data_agents.exists():
        shutil.copy(defaults_agents, data_agents)

    # Copiar prompts por defecto
    defaults_prompts = DEFAULTS_DIR / "prompts"
    data_prompts = DATA_DIR / "prompts" / "system"
    if defaults_prompts.exists():
        for prompt_file in defaults_prompts.glob("*.md"):
            dest = data_prompts / prompt_file.name
            if not dest.exists():
                shutil.copy(prompt_file, dest)

    # Inicializar config global si no existe
    config_file = DATA_DIR / "config.json"
    if not config_file.exists():
        config = {
            "version": "0.1.0",
            "created_at": datetime.utcnow().isoformat(),
            "default_model": "llama3.2:3b",
            "ollama_url": OLLAMA_URL,
            "language": "es",
        }
        config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    # Verificar modelos disponibles en Ollama y actualizar config si es necesario
    await _verify_and_fix_models()

    logger.info(f"CSS Brand Assistant iniciado. DATA_DIR={DATA_DIR}")


async def _verify_and_fix_models():
    """Verifica que el modelo configurado existe en Ollama.
    Si no existe, intenta descargarlo automáticamente (ollama pull).
    Si Ollama no está disponible, lo ignora silenciosamente.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            logger.warning("Ollama no disponible al inicio — se verificará cuando el usuario lo necesite")
            return

        available_models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"Modelos disponibles en Ollama: {available_models}")

        # Leer modelo configurado
        config_file = DATA_DIR / "config.json"
        config = load_json(config_file, {})
        configured_model = config.get("default_model", "llama3.2:3b")

        # Verificar si el modelo configurado está disponible
        model_base = configured_model.split(":")[0]
        model_found = any(
            m == configured_model or m.startswith(model_base + ":")
            for m in available_models
        )

        if model_found:
            logger.info(f"Modelo configurado '{configured_model}' disponible ✓")
            return

        # El modelo no está disponible → intentar descargarlo en background
        logger.info(
            f"Modelo '{configured_model}' no encontrado en Ollama. "
            f"Iniciando descarga automática en background..."
        )
        _start_pull_background(configured_model)

    except Exception as e:
        logger.warning(f"No se pudo verificar modelos al inicio: {e}")


def _is_model_available(model: str) -> bool:
    """Comprueba rápidamente si un modelo ya está descargado en Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        available = [m["name"] for m in resp.json().get("models", [])]
        model_base = model.split(":")[0]
        return any(
            m == model or m.startswith(model_base + ":")
            for m in available
        )
    except Exception:
        return False


def _start_pull_background(model: str) -> None:
    """Lanza la descarga de un modelo Ollama en un hilo background.
    Actualiza _pull_status con el progreso para que el frontend pueda consultarlo.
    Si ya hay una descarga en curso para ese modelo, no lanza otra.
    """
    with _pull_lock:
        existing = _pull_status.get(model, {})
        if existing.get("status") in ("pulling", "queued"):
            logger.info(f"Descarga de '{model}' ya en curso, no se lanza otra")
            return
        _pull_status[model] = {
            "status": "queued",
            "progress": 0,
            "error": None,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }

    def _do_pull():
        with _pull_lock:
            _pull_status[model]["status"] = "pulling"
        logger.info(f"[pull] Iniciando descarga de modelo: {model}")
        try:
            # Usar stream=True para poder reportar progreso
            pull_resp = requests.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": model, "stream": True},
                stream=True,
                timeout=1800,  # 30 minutos máximo
            )
            pull_resp.raise_for_status()

            last_progress = 0
            for raw_line in pull_resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                status_msg = chunk.get("status", "")
                total = chunk.get("total", 0)
                completed = chunk.get("completed", 0)

                if total and total > 0:
                    pct = int(completed * 100 / total)
                    if pct != last_progress:
                        last_progress = pct
                        with _pull_lock:
                            _pull_status[model]["progress"] = pct
                            _pull_status[model]["status_msg"] = status_msg

                if "error" in chunk:
                    raise RuntimeError(chunk["error"])

                if status_msg == "success":
                    break

            with _pull_lock:
                _pull_status[model]["status"] = "done"
                _pull_status[model]["progress"] = 100
                _pull_status[model]["completed_at"] = datetime.utcnow().isoformat()
            logger.info(f"[pull] Modelo '{model}' descargado correctamente")

        except Exception as exc:
            with _pull_lock:
                _pull_status[model]["status"] = "error"
                _pull_status[model]["error"] = str(exc)
                _pull_status[model]["completed_at"] = datetime.utcnow().isoformat()
            logger.error(f"[pull] Error descargando modelo '{model}': {exc}")

    t = threading.Thread(target=_do_pull, daemon=True, name=f"pull-{model}")
    t.start()


# ---------------------------------------------------------------------------
# Utilidades de persistencia
# ---------------------------------------------------------------------------
def save_json(path: Path, data: Any) -> None:
    """Guarda datos como JSON con formato legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # encoding='utf-8' es obligatorio en Windows donde el default es cp1252
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json(path: Path, default=None) -> Any:
    """Carga JSON desde disco, retorna default si no existe."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default if default is not None else {}


def get_system_prompt(agent_id: str) -> str:
    """Lee el system prompt de un agente desde disco."""
    prompt_file = DATA_DIR / "prompts" / "system" / f"{agent_id}.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    # Fallback: buscar en agents.json
    agents = load_json(DATA_DIR / "agents" / "agents.json", {})
    for agent in agents.get("agents", []):
        if agent.get("id") == agent_id:
            return agent.get("system_prompt", "")
    return ""


def log_audit(agent_id: str, task: str, inputs: dict, output: str,
              model: str, latency_ms: int, success: bool, error: str = "") -> None:
    """Registra una entrada de auditoría para trazabilidad de agentes."""
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "agent_id": agent_id,
        "task": task,
        "model": model,
        "inputs_summary": str(inputs)[:500],
        "output_summary": output[:500] if output else "",
        "latency_ms": latency_ms,
        "success": success,
        "error": error,
    }
    audit_file = DATA_DIR / "audit" / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Integración con Ollama
# ---------------------------------------------------------------------------
# Cache para recordar qué endpoint de Ollama funciona (evita reintentos en cada llamada)
_ollama_api_endpoint: Optional[str] = None


def get_ollama_timeout(kind: str = "default") -> int:
    """Retorna el timeout configurado para llamadas a Ollama.

    Jerarquía de precedencia (mayor a menor):
      1. config.json  (clave: ollama_timeout / ollama_timeout_campaign / ollama_timeout_adn)
      2. Variables de entorno OLLAMA_TIMEOUT / OLLAMA_TIMEOUT_CAMPAIGN / OLLAMA_TIMEOUT_ADN
      3. Valores por defecto del código (300s / 600s / 300s)

    Args:
        kind: "default" | "campaign" | "adn"
    """
    config = load_json(DATA_DIR / "config.json", {})
    key_map = {
        "default":  ("ollama_timeout",          OLLAMA_TIMEOUT_DEFAULT),
        "campaign": ("ollama_timeout_campaign",  OLLAMA_TIMEOUT_CAMPAIGN),
        "adn":      ("ollama_timeout_adn",       OLLAMA_TIMEOUT_ADN),
    }
    config_key, fallback = key_map.get(kind, ("ollama_timeout", OLLAMA_TIMEOUT_DEFAULT))
    try:
        return int(config.get(config_key, fallback))
    except (ValueError, TypeError):
        return fallback


def _fix_encoding(text: str) -> str:
    """Repara caracteres mal codificados en respuestas del LLM.

    En Windows con Ollama antiguo, la respuesta HTTP puede llegar con
    encoding incorrecto (latin-1 detectado en lugar de UTF-8), produciendo
    secuencias como 'Ã\xa1' en lugar de 'á', 'Ã\xb1' en lugar de 'ñ', etc.

    Este helper intenta detectar y corregir esa doble-codificación.
    """
    if not text:
        return text
    try:
        # Detectar si el texto tiene caracteres que parecen UTF-8 mal interpretados como latin-1
        # Patrón: caracteres en rango 0xC0-0xFF seguidos de caracteres en rango 0x80-0xBF
        # (característico de UTF-8 de 2 bytes interpretado como latin-1)
        fixed = text.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
        # Solo usar la versión reparada si tiene menos caracteres de reemplazo que el original
        if fixed.count("�") < text.count("�") and fixed != text:
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def call_ollama(model: str, system_prompt: str, user_message: str,
                temperature: float = 0.7, timeout: Optional[int] = None) -> str:
    """Llama al LLM local vía Ollama API.
    
    Detecta automáticamente si Ollama soporta /api/chat (v0.1.14+) o solo
    /api/generate (versiones antiguas, común en Windows con winget).
    """
    global _ollama_api_endpoint

    # Resolver timeout: si no se pasó explícitamente, leer de config/env
    if timeout is None:
        timeout = get_ollama_timeout("default")

    try:
        # --- Intento 1: /api/chat (Ollama moderno, v0.1.14+) ---
        if _ollama_api_endpoint in (None, "chat"):
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_message},
                        ],
                        "options": {"temperature": temperature},
                        "stream": False,
                    },
                    timeout=timeout,
                )
                if response.status_code == 404:
                    # Puede ser que /api/chat no exista (Ollama antiguo)
                    # o que el modelo no esté descargado.
                    # Distinguimos por el body de la respuesta.
                    try:
                        err_body = response.json()
                        err_msg = err_body.get("error", "").lower()
                    except Exception:
                        err_msg = ""

                    if "model" in err_msg and ("not found" in err_msg or "pull" in err_msg):
                        # El modelo no está descargado → auto-pull
                        logger.warning(f"Ollama /api/chat: modelo '{model}' no encontrado. Iniciando descarga...")
                        _start_pull_background(model)
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                f"El modelo '{model}' no está descargado todavía. "
                                f"Se ha iniciado la descarga automática en background. "
                                f"Consulta el estado en Agentes IA o espera unos minutos e intenta de nuevo."
                            )
                        )
                    else:
                        # /api/chat no existe en esta versión de Ollama → fallback a /api/generate
                        logger.warning("Ollama: /api/chat devolvió 404, usando /api/generate como fallback")
                        _ollama_api_endpoint = "generate"
                else:
                    response.raise_for_status()
                    _ollama_api_endpoint = "chat"
                    # Forzar UTF-8 en la decodificación de la respuesta HTTP
                    # (en Windows, requests puede detectar incorrectamente latin-1)
                    response.encoding = "utf-8"
                    content = response.json()["message"]["content"]
                    # Reparar caracteres mal codificados (latin-1 interpretado como UTF-8)
                    content = _fix_encoding(content)
                    return content
            except HTTPException:
                raise
            except requests.exceptions.HTTPError as e:
                if "404" in str(e):
                    logger.warning("Ollama: /api/chat no disponible, usando /api/generate")
                    _ollama_api_endpoint = "generate"
                else:
                    raise

        # --- Intento 2: /api/generate (Ollama antiguo, Windows winget) ---
        if _ollama_api_endpoint == "generate":
            # Combinar system prompt y user message en un solo prompt
            full_prompt = f"{system_prompt}\n\nUsuario: {user_message}\n\nRespuesta:"
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "options": {"temperature": temperature},
                    "stream": False,
                },
                timeout=timeout,
            )
            if response.status_code == 404:
                # 404 en /api/generate = modelo no descargado
                logger.warning(f"Modelo '{model}' no encontrado en Ollama. Iniciando descarga automática...")
                _start_pull_background(model)
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"El modelo '{model}' no está descargado todavía. "
                        f"Se ha iniciado la descarga automática en background. "
                        f"Consulta el estado en Agentes IA o espera unos minutos e intenta de nuevo."
                    )
                )
            response.raise_for_status()
            # Forzar UTF-8 en la decodificación de la respuesta HTTP
            response.encoding = "utf-8"
            content = response.json().get("response", "")
            # Reparar caracteres mal codificados
            content = _fix_encoding(content)
            return content

        raise HTTPException(status_code=503, detail="No se pudo determinar el endpoint de Ollama")

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama no está disponible. Asegúrate de que Ollama esté corriendo.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al llamar a Ollama: {str(e)}")


def get_active_model() -> str:
    """Retorna el modelo activo configurado."""
    config = load_json(DATA_DIR / "config.json", {})
    return config.get("default_model", "llama3.2:3b")


# ---------------------------------------------------------------------------
# Modelos Pydantic
# ---------------------------------------------------------------------------
class BrandCreate(BaseModel):
    name: str
    website: Optional[str] = None
    description: Optional[str] = None
    sector: Optional[str] = None
    target_markets: Optional[str] = None
    language: str = "es"


class BrandUpdate(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    sector: Optional[str] = None
    target_markets: Optional[str] = None
    language: Optional[str] = None


class ADNUpdate(BaseModel):
    field: str
    value: Any
    reason: Optional[str] = "Edición manual"


class InterviewMessage(BaseModel):
    brand_id: str
    session_id: Optional[str] = None
    message: str


class CampaignCreate(BaseModel):
    brand_id: str
    adn_version: Optional[str] = None
    name: str
    objective: str
    secondary_objective: Optional[str] = None
    product_or_topic: str
    target_audience: str
    start_date: str
    end_date: str
    channels: List[str]
    frequency: str = "diaria"
    budget: Optional[float] = None
    restrictions: Optional[str] = None


class PublicationUpdate(BaseModel):
    text: Optional[str] = None
    hashtags: Optional[List[str]] = None
    cta: Optional[str] = None
    image_prompt: Optional[str] = None
    scheduled_at: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class AgentConfigUpdate(BaseModel):
    agent_id: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None


class WebsiteAnalyzeRequest(BaseModel):
    brand_id: str
    url: str


# ---------------------------------------------------------------------------
# RUTAS: Sistema
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Verificación de estado del servidor."""
    return {"status": "ok", "version": "0.1.0", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/config")
def get_config():
    """Retorna la configuración global del plugin."""
    return load_json(DATA_DIR / "config.json", {})


@app.put("/api/config")
def update_config(updates: dict):
    """Actualiza la configuración global."""
    config = load_json(DATA_DIR / "config.json", {})
    config.update(updates)
    save_json(DATA_DIR / "config.json", config)
    return config


@app.get("/api/ollama/status")
def ollama_status():
    """Verifica si Ollama está disponible y lista los modelos instalados."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"available": True, "models": models}
    except Exception:
        return {"available": False, "models": []}


@app.post("/api/models/pull")
def pull_model_endpoint(body: dict):
    """Inicia la descarga de un modelo Ollama en background.
    Body: {"model": "llama3.2:3b"}
    Retorna inmediatamente con el estado inicial de la descarga.
    """
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Se requiere el campo 'model'")

    # Si ya está disponible, no hace falta descargar
    if _is_model_available(model):
        return {
            "model": model,
            "status": "already_available",
            "message": f"El modelo '{model}' ya está disponible en Ollama.",
        }

    # Iniciar descarga en background
    _start_pull_background(model)
    with _pull_lock:
        status_info = _pull_status.get(model, {})

    return {
        "model": model,
        "status": status_info.get("status", "queued"),
        "message": f"Descarga de '{model}' iniciada en background. Consulta /api/models/pull/{model}/status para el progreso.",
        "pull_info": status_info,
    }


@app.get("/api/models/pull/{model_name:path}/status")
def pull_model_status_endpoint(model_name: str):
    """Consulta el estado de la descarga de un modelo.
    Retorna: {status: queued|pulling|done|error, progress: 0-100, error: str|null}
    """
    with _pull_lock:
        info = _pull_status.get(model_name, None)

    if info is None:
        # Verificar si ya está disponible aunque no haya registro de pull
        if _is_model_available(model_name):
            return {
                "model": model_name,
                "status": "available",
                "progress": 100,
                "error": None,
                "message": f"El modelo '{model_name}' está disponible.",
            }
        return {
            "model": model_name,
            "status": "not_started",
            "progress": 0,
            "error": None,
            "message": f"No hay descarga en curso para '{model_name}'.",
        }

    return {
        "model": model_name,
        "status": info.get("status"),
        "progress": info.get("progress", 0),
        "status_msg": info.get("status_msg", ""),
        "error": info.get("error"),
        "started_at": info.get("started_at"),
        "completed_at": info.get("completed_at"),
    }


@app.get("/api/models/pull/all")
def pull_all_status():
    """Retorna el estado de todas las descargas de modelos en curso o completadas."""
    with _pull_lock:
        return {"pulls": dict(_pull_status)}


# ---------------------------------------------------------------------------
# RUTAS: Marcas
# ---------------------------------------------------------------------------
@app.get("/api/brands")
def list_brands():
    """Lista todas las marcas registradas."""
    brands_dir = DATA_DIR / "brands"
    brands = []
    for brand_file in brands_dir.glob("*/brand.json"):
        brand = load_json(brand_file)
        if brand:
            brands.append(brand)
    brands.sort(key=lambda b: b.get("created_at", ""), reverse=True)
    return {"brands": brands}


@app.post("/api/brands", status_code=201)
def create_brand(brand: BrandCreate):
    """Crea una nueva marca."""
    brand_id = str(uuid.uuid4())
    brand_dir = DATA_DIR / "brands" / brand_id
    brand_dir.mkdir(parents=True, exist_ok=True)

    brand_data = {
        "id": brand_id,
        "name": brand.name,
        "website": brand.website,
        "description": brand.description,
        "sector": brand.sector,
        "target_markets": brand.target_markets,
        "language": brand.language,
        "onboarding_status": "pending",   # pending | analyzing | interviewing | complete
        "adn_version": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    save_json(brand_dir / "brand.json", brand_data)
    logger.info(f"Marca creada: {brand_id} — {brand.name}")
    return brand_data


@app.get("/api/brands/{brand_id}")
def get_brand(brand_id: str):
    """Retorna los datos de una marca específica."""
    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"
    brand = load_json(brand_file)
    if not brand:
        raise HTTPException(status_code=404, detail="Marca no encontrada")
    return brand


@app.put("/api/brands/{brand_id}")
def update_brand(brand_id: str, updates: BrandUpdate):
    """Actualiza los datos de una marca."""
    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"
    brand = load_json(brand_file)
    if not brand:
        raise HTTPException(status_code=404, detail="Marca no encontrada")

    update_data = updates.dict(exclude_none=True)
    brand.update(update_data)
    brand["updated_at"] = datetime.utcnow().isoformat()
    save_json(brand_file, brand)
    return brand


@app.delete("/api/brands/{brand_id}")
def delete_brand(brand_id: str):
    """Elimina una marca y todos sus datos asociados."""
    brand_dir = DATA_DIR / "brands" / brand_id
    if not brand_dir.exists():
        raise HTTPException(status_code=404, detail="Marca no encontrada")
    shutil.rmtree(brand_dir)
    return {"message": "Marca eliminada correctamente"}


# ---------------------------------------------------------------------------
# RUTAS: Análisis de sitio web
# ---------------------------------------------------------------------------
@app.post("/api/brands/{brand_id}/analyze-website")
async def analyze_website(brand_id: str, request: WebsiteAnalyzeRequest,
                           background_tasks: BackgroundTasks):
    """
    Inicia el análisis del sitio web de la marca.
    El análisis se ejecuta en background y actualiza el ADN borrador.
    """
    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"
    brand = load_json(brand_file)
    if not brand:
        raise HTTPException(status_code=404, detail="Marca no encontrada")

    # Actualizar estado
    brand["onboarding_status"] = "analyzing"
    brand["website"] = request.url
    brand["updated_at"] = datetime.utcnow().isoformat()
    save_json(brand_file, brand)

    background_tasks.add_task(_analyze_website_task, brand_id, request.url)
    return {"message": "Análisis iniciado", "brand_id": brand_id, "status": "analyzing"}


async def _analyze_website_task(brand_id: str, url: str):
    """Tarea de análisis de sitio web en background."""
    import time
    start = time.time()
    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"

    try:
        # Extraer texto del sitio web
        website_text = _scrape_website(url)

        # Llamar al agente analizador de marca
        model = get_active_model()
        system_prompt = get_system_prompt("brand_analyzer")
        if not system_prompt:
            system_prompt = _get_default_brand_analyzer_prompt()

        user_message = f"""Analiza el siguiente sitio web y extrae las señales de identidad de marca.
URL: {url}

CONTENIDO DEL SITIO:
{website_text[:4000]}

Responde en formato JSON con los campos del ADN empresarial."""

        result = call_ollama(
            model, system_prompt, user_message,
            temperature=0.3,
            timeout=get_ollama_timeout("adn"),
        )
        latency = int((time.time() - start) * 1000)

        # Intentar parsear JSON del resultado
        adn_draft = _parse_adn_from_llm(result, url)

        # Guardar borrador de ADN
        adn_id = str(uuid.uuid4())
        adn_data = {
            "id": adn_id,
            "brand_id": brand_id,
            "version": "0.1-draft",
            "status": "draft",
            "source": "website_analysis",
            "website_url": url,
            "created_at": datetime.utcnow().isoformat(),
            "fields": adn_draft,
            "raw_llm_output": result,
        }
        save_json(DATA_DIR / "brands" / brand_id / "adn_draft.json", adn_data)

        # Actualizar estado de la marca
        brand = load_json(brand_file)
        brand["onboarding_status"] = "interviewing"
        brand["adn_draft_id"] = adn_id
        brand["updated_at"] = datetime.utcnow().isoformat()
        save_json(brand_file, brand)

        log_audit("brand_analyzer", "analyze_website",
                  {"brand_id": brand_id, "url": url},
                  result, model, latency, True)
        logger.info(f"Análisis completado para marca {brand_id}")

    except Exception as e:
        brand = load_json(brand_file)
        brand["onboarding_status"] = "error"
        brand["error"] = str(e)
        brand["updated_at"] = datetime.utcnow().isoformat()
        save_json(brand_file, brand)
        log_audit("brand_analyzer", "analyze_website",
                  {"brand_id": brand_id, "url": url},
                  "", get_active_model(), 0, False, str(e))
        logger.error(f"Error en análisis de marca {brand_id}: {e}")


def _scrape_website(url: str) -> str:
    """Extrae texto visible de un sitio web."""
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CSSBrandAssistant/0.1)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        # Forzar UTF-8 para evitar problemas de codificacion en Windows (cp1252)
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding="utf-8")

        # Eliminar scripts y estilos
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extraer texto significativo
        texts = []
        for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "span", "a"]):
            text = tag.get_text(strip=True)
            if len(text) > 20:
                texts.append(text)

        return "\n".join(texts[:200])
    except Exception as e:
        return f"[Error al acceder al sitio: {str(e)}]"


def _parse_adn_from_llm(llm_output: str, url: str) -> dict:
    """Intenta parsear JSON del output del LLM, con fallback a estructura básica."""
    import re
    # Buscar bloque JSON en el output
    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: estructura básica inferida del texto
    return {
        "value_proposition": "",
        "sector": "",
        "tone": "",
        "personality_traits": [],
        "color_palette": [],
        "typography": "",
        "visual_style": "",
        "products_services": [],
        "brand_promises": [],
        "target_audience": "",
        "formality_level": "medium",
        "raw_analysis": llm_output[:2000],
        "source_url": url,
    }


def _get_default_brand_analyzer_prompt() -> str:
    return """Eres un experto en branding y marketing digital especializado en análisis de identidad de marca.
Tu tarea es analizar el contenido de un sitio web y extraer señales de identidad de marca.

Debes identificar y estructurar en formato JSON:
- value_proposition: propuesta de valor aparente
- sector: sector o categoría de negocio
- tone: tono comunicacional dominante (formal/informal/técnico/emocional/etc)
- personality_traits: rasgos de personalidad de marca (lista)
- color_palette: paleta de colores mencionada o inferida
- typography: estilo tipográfico detectado
- visual_style: estilo visual predominante
- products_services: tipos de productos o servicios principales (lista)
- brand_promises: promesas de marca repetidas (lista)
- target_audience: público objetivo sugerido
- formality_level: nivel de formalidad (low/medium/high)
- differentiators: diferenciadores competitivos detectados (lista)
- content_themes: temas frecuentes de contenido (lista)

Responde ÚNICAMENTE con el JSON, sin texto adicional."""


# ---------------------------------------------------------------------------
# RUTAS: ADN Empresarial
# ---------------------------------------------------------------------------
@app.get("/api/brands/{brand_id}/adn")
def get_adn(brand_id: str):
    """Retorna el ADN activo de una marca (draft o aprobado)."""
    # Primero buscar ADN aprobado
    adn_file = DATA_DIR / "brands" / brand_id / "adn.json"
    if adn_file.exists():
        return load_json(adn_file)
    # Fallback: borrador
    draft_file = DATA_DIR / "brands" / brand_id / "adn_draft.json"
    if draft_file.exists():
        return load_json(draft_file)
    raise HTTPException(status_code=404, detail="ADN no encontrado. Primero analiza el sitio web.")


@app.get("/api/brands/{brand_id}/adn/versions")
def get_adn_versions(brand_id: str):
    """Lista todas las versiones del ADN de una marca."""
    versions_dir = DATA_DIR / "brands" / brand_id / "adn_versions"
    versions = []
    if versions_dir.exists():
        for v_file in versions_dir.glob("*.json"):
            v = load_json(v_file)
            if v:
                versions.append({"id": v.get("id"), "version": v.get("version"),
                                  "status": v.get("status"), "created_at": v.get("created_at")})
    versions.sort(key=lambda v: v.get("created_at", ""), reverse=True)
    return {"versions": versions}


@app.put("/api/brands/{brand_id}/adn/field")
def update_adn_field(brand_id: str, update: ADNUpdate):
    """Actualiza un campo específico del ADN."""
    adn_file = DATA_DIR / "brands" / brand_id / "adn_draft.json"
    if not adn_file.exists():
        adn_file = DATA_DIR / "brands" / brand_id / "adn.json"
    adn = load_json(adn_file)
    if not adn:
        raise HTTPException(status_code=404, detail="ADN no encontrado")

    adn["fields"][update.field] = update.value
    adn["last_edited_at"] = datetime.utcnow().isoformat()
    adn["edit_history"] = adn.get("edit_history", [])
    adn["edit_history"].append({
        "field": update.field,
        "value": update.value,
        "reason": update.reason,
        "timestamp": datetime.utcnow().isoformat(),
    })
    save_json(adn_file, adn)
    return adn


@app.post("/api/brands/{brand_id}/adn/approve")
def approve_adn(brand_id: str):
    """Aprueba el borrador de ADN y crea una versión oficial."""
    draft_file = DATA_DIR / "brands" / brand_id / "adn_draft.json"
    draft = load_json(draft_file)
    if not draft:
        raise HTTPException(status_code=404, detail="No hay borrador de ADN para aprobar")

    # Versionar el ADN anterior si existe
    current_adn_file = DATA_DIR / "brands" / brand_id / "adn.json"
    if current_adn_file.exists():
        current = load_json(current_adn_file)
        versions_dir = DATA_DIR / "brands" / brand_id / "adn_versions"
        versions_dir.mkdir(exist_ok=True)
        save_json(versions_dir / f"{current['id']}.json", current)

    # Aprobar el borrador
    draft["status"] = "approved"
    draft["approved_at"] = datetime.utcnow().isoformat()
    version_num = len(list((DATA_DIR / "brands" / brand_id / "adn_versions").glob("*.json"))) + 1
    draft["version"] = f"{version_num}.0"
    save_json(current_adn_file, draft)

    # Actualizar estado de la marca
    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"
    brand = load_json(brand_file)
    brand["onboarding_status"] = "complete"
    brand["adn_version"] = draft["version"]
    brand["updated_at"] = datetime.utcnow().isoformat()
    save_json(brand_file, brand)

    return {"message": "ADN aprobado", "version": draft["version"], "adn": draft}


# ---------------------------------------------------------------------------
# RUTAS: Entrevista guiada por agente
# ---------------------------------------------------------------------------
@app.post("/api/brands/{brand_id}/interview")
async def interview_agent(brand_id: str, msg: InterviewMessage):
    """
    Conduce la entrevista de descubrimiento de marca con el agente entrevistador.
    Mantiene historial de sesión y actualiza el ADN incrementalmente.
    """
    import time
    start = time.time()

    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"
    brand = load_json(brand_file)
    if not brand:
        raise HTTPException(status_code=404, detail="Marca no encontrada")

    # Gestionar sesión
    session_id = msg.session_id or str(uuid.uuid4())
    session_file = DATA_DIR / "sessions" / f"{brand_id}_{session_id}.json"
    session = load_json(session_file, {"id": session_id, "brand_id": brand_id,
                                        "messages": [], "created_at": datetime.utcnow().isoformat()})

    # Cargar ADN borrador como contexto
    adn_draft = load_json(DATA_DIR / "brands" / brand_id / "adn_draft.json", {})
    adn_context = json.dumps(adn_draft.get("fields", {}), ensure_ascii=False)[:2000]

    # Construir historial de conversación
    history = session.get("messages", [])
    history_text = "\n".join([
        f"{'Usuario' if m['role'] == 'user' else 'Agente'}: {m['content']}"
        for m in history[-10:]  # últimos 10 mensajes
    ])

    model = get_active_model()
    system_prompt = get_system_prompt("brand_interviewer")
    if not system_prompt:
        system_prompt = _get_default_interviewer_prompt()

    user_message = f"""CONTEXTO DEL ADN ACTUAL:
{adn_context}

HISTORIAL DE CONVERSACIÓN:
{history_text}

MENSAJE DEL USUARIO: {msg.message}"""

    response = call_ollama(
        model, system_prompt, user_message,
        temperature=0.7,
        timeout=get_ollama_timeout("default"),
    )
    latency = int((time.time() - start) * 1000)

    # Guardar mensajes en sesión
    history.append({"role": "user", "content": msg.message, "timestamp": datetime.utcnow().isoformat()})
    history.append({"role": "assistant", "content": response, "timestamp": datetime.utcnow().isoformat()})
    session["messages"] = history
    session["updated_at"] = datetime.utcnow().isoformat()
    save_json(session_file, session)

    log_audit("brand_interviewer", "interview",
              {"brand_id": brand_id, "session_id": session_id},
              response, model, latency, True)

    return {
        "session_id": session_id,
        "response": response,
        "message_count": len(history),
    }


@app.post("/api/brands/{brand_id}/interview/finish")
async def finish_interview(brand_id: str, session_id: Optional[str] = None):
    """
    Finaliza la entrevista de descubrimiento y dispara la actualización del ADN
    con todos los insights recopilados en la sesión.
    """
    import time
    start = time.time()

    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"
    brand = load_json(brand_file)
    if not brand:
        raise HTTPException(status_code=404, detail="Marca no encontrada")

    # Recopilar todos los mensajes de la sesión
    session_messages = []
    if session_id:
        session_file = DATA_DIR / "sessions" / f"{brand_id}_{session_id}.json"
        session = load_json(session_file, {})
        session_messages = session.get("messages", [])
    else:
        # Buscar la sesión más reciente de esta marca
        sessions_dir = DATA_DIR / "sessions"
        brand_sessions = sorted(
            sessions_dir.glob(f"{brand_id}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if brand_sessions:
            session = load_json(brand_sessions[0], {})
            session_messages = session.get("messages", [])
            session_id = session.get("id", "unknown")

    if not session_messages:
        return {"status": "no_session", "message": "No se encontró sesión activa para esta marca"}

    # Construir transcript completo
    transcript = "\n".join([
        f"{'Usuario' if m['role'] == 'user' else 'Agente'}: {m['content']}"
        for m in session_messages
    ])

    # Cargar ADN borrador actual
    adn_draft = load_json(DATA_DIR / "brands" / brand_id / "adn_draft.json", {})
    adn_current = json.dumps(adn_draft.get("fields", {}), ensure_ascii=False)[:2000]

    # Usar el agente analizador para actualizar el ADN con los insights de la entrevista
    model = get_active_model()
    system_prompt = get_system_prompt("brand_analyzer") or _get_default_analyzer_prompt()

    user_message = (
        f"Analiza la siguiente entrevista de descubrimiento de marca y actualiza el ADN empresarial.\n\n"
        f"ADN ACTUAL:\n{adn_current}\n\n"
        f"TRANSCRIPT DE LA ENTREVISTA:\n{transcript[:4000]}\n\n"
        f"Extrae todos los insights relevantes y devuelve el ADN actualizado en JSON válido.\n"
        f"Usa los mismos campos del ADN actual y agrega información nueva encontrada en la entrevista."
    )

    try:
        result = call_ollama(
            model, system_prompt, user_message,
            temperature=0.3,
            timeout=get_ollama_timeout("adn"),
        )
        latency = int((time.time() - start) * 1000)

        parsed = _parse_llm_json(result)
        if parsed:
            # Normalizar nombres de campo: el LLM puede devolver nombres en inglés
            # que deben mapearse a los campos canónicos usados en la UI
            FIELD_ALIASES = {
                # Inglés → canónico
                "value_proposition": "value_proposition",
                "sector": "sector",
                "tone": "tone",
                "personality_traits": "personality_traits",
                "color_palette": "color_palette",
                "typography": "typography",
                "visual_style": "visual_style",
                "products_services": "products_services",
                "brand_promises": "brand_promises",
                "target_audience": "target_audience",
                "formality_level": "formality_level",
                "differentiators": "differentiators",
                "content_themes": "content_themes",
                # Español → canónico
                "propuesta_de_valor": "value_proposition",
                "propuesta_valor": "value_proposition",
                "tono": "tone",
                "tono_comunicacional": "tone",
                "personalidad": "personality_traits",
                "personalidad_de_marca": "personality_traits",
                "rasgos_personalidad": "personality_traits",
                "paleta_colores": "color_palette",
                "paleta_de_colores": "color_palette",
                "tipografia": "typography",
                "estilo_visual": "visual_style",
                "productos_servicios": "products_services",
                "productos": "products_services",
                "promesas": "brand_promises",
                "promesas_de_marca": "brand_promises",
                "publico_objetivo": "target_audience",
                "audiencia": "target_audience",
                "nivel_formalidad": "formality_level",
                "diferenciadores": "differentiators",
                "temas_contenido": "content_themes",
                "temas_de_contenido": "content_themes",
            }

            current_fields = adn_draft.get("fields", {})
            for raw_key, value in parsed.items():
                if not value:
                    continue
                # Normalizar clave
                canonical = FIELD_ALIASES.get(raw_key.lower(), raw_key.lower())
                current_fields[canonical] = value

            adn_draft["fields"] = current_fields
            adn_draft["updated_at"] = datetime.utcnow().isoformat()
            adn_draft["interview_session_id"] = session_id
            save_json(DATA_DIR / "brands" / brand_id / "adn_draft.json", adn_draft)

            # Actualizar estado de la marca a 'complete' (entrevista finalizada)
            brand["onboarding_status"] = "complete"
            brand["updated_at"] = datetime.utcnow().isoformat()
            save_json(brand_file, brand)

            log_audit("brand_analyzer", "finish_interview",
                      {"brand_id": brand_id, "session_id": session_id, "messages": len(session_messages),
                       "fields_updated": list(current_fields.keys())},
                      result[:500], model, latency, True)

            return {
                "status": "finished",
                "message": "Entrevista finalizada. El ADN ha sido actualizado con los insights recopilados.",
                "adn_updated": True,
                "message_count": len(session_messages),
                "fields_updated": len(current_fields),
            }
        else:
            # Aunque no se parseó JSON, guardar el texto crudo en raw_analysis
            current_fields = adn_draft.get("fields", {})
            current_fields["raw_analysis"] = result[:3000]
            adn_draft["fields"] = current_fields
            adn_draft["updated_at"] = datetime.utcnow().isoformat()
            save_json(DATA_DIR / "brands" / brand_id / "adn_draft.json", adn_draft)

            # Cambiar estado a complete de todas formas
            brand["onboarding_status"] = "complete"
            brand["updated_at"] = datetime.utcnow().isoformat()
            save_json(brand_file, brand)

            return {
                "status": "finished",
                "message": "Entrevista finalizada. El análisis se guardó como texto sin estructurar.",
                "adn_updated": False,
                "message_count": len(session_messages),
            }
    except Exception as e:
        logger.error(f"Error al finalizar entrevista: {e}")
        return {
            "status": "finished",
            "message": f"Entrevista finalizada (con error al actualizar ADN: {str(e)})",
            "adn_updated": False,
            "message_count": len(session_messages),
        }


def _get_default_interviewer_prompt() -> str:
    return """Eres un consultor experto en marketing y branding con 20 años de experiencia.
Tu rol es conducir una entrevista de descubrimiento de marca para una PYME.

OBJETIVO: Completar y refinar el ADN empresarial de la marca mediante preguntas inteligentes y contextuales.

ESTILO DE ENTREVISTA:
- Actúa como consultor, no como formulario
- Haz UNA sola pregunta a la vez, bien formulada
- Ancla cada pregunta a lo que ya sabes del ADN
- Sé empático, claro y profesional
- Usa español neutro y accesible

BLOQUES TEMÁTICOS A CUBRIR:
1. Identidad y posicionamiento
2. Cliente ideal y audiencia
3. Tono y personalidad de marca
4. Propuesta de valor diferencial
5. Restricciones y límites de comunicación
6. Objetivos de marketing

Cuando el usuario responda, extrae insights relevantes y formula la siguiente pregunta lógica.
Si el ADN ya tiene información sobre un tema, profundiza en lugar de repetir preguntas básicas."""


# ---------------------------------------------------------------------------
# RUTAS: Campañas
# ---------------------------------------------------------------------------
@app.get("/api/brands/{brand_id}/campaigns")
def list_campaigns(brand_id: str):
    """Lista todas las campañas de una marca."""
    campaigns_dir = DATA_DIR / "campaigns"
    campaigns = []
    for camp_file in campaigns_dir.glob(f"{brand_id}_*/campaign.json"):
        camp = load_json(camp_file)
        if camp:
            campaigns.append(camp)
    campaigns.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return {"campaigns": campaigns}


@app.get("/api/campaigns/{campaign_id}/progress")
def get_campaign_progress(campaign_id: str):
    """Retorna el progreso de generación de una campaña en tiempo real.

    Respuesta:
    - status: generating | active | error | paused
    - publications_done, publications_total, pct: progreso global
    - channels: lista de {channel, done, total, pct} por canal
    """
    campaigns_dir = DATA_DIR / "campaigns"
    for camp_dir in campaigns_dir.iterdir():
        if camp_dir.is_dir() and campaign_id in camp_dir.name:
            camp = load_json(camp_dir / "campaign.json", {})
            if camp.get("id") == campaign_id:
                progress = camp.get("generation_progress", {})
                return {
                    "campaign_id": campaign_id,
                    "status": camp.get("status", "unknown"),
                    "publications_count": camp.get("publications_count", 0),
                    "publications_done": progress.get("publications_done", 0),
                    "publications_total": progress.get("publications_total", 0),
                    "pct": progress.get("pct", 0),
                    "batch": progress.get("batch", 0),
                    "total_batches": progress.get("total_batches", 0),
                    "channels": progress.get("channels", []),
                    "error": camp.get("error"),
                    "updated_at": camp.get("updated_at"),
                }
    raise HTTPException(status_code=404, detail="Campaña no encontrada")


@app.post("/api/brands/{brand_id}/campaigns", status_code=201)
async def create_campaign(brand_id: str, campaign: CampaignCreate,
                           background_tasks: BackgroundTasks):
    """Crea una nueva campaña y genera la planificación temporal en background."""
    brand_file = DATA_DIR / "brands" / brand_id / "brand.json"
    brand = load_json(brand_file)
    if not brand:
        raise HTTPException(status_code=404, detail="Marca no encontrada")

    # Verificar que existe ADN
    adn = load_json(DATA_DIR / "brands" / brand_id / "adn.json") or \
          load_json(DATA_DIR / "brands" / brand_id / "adn_draft.json")
    if not adn:
        raise HTTPException(status_code=400,
                            detail="La marca necesita un ADN antes de crear campañas")

    campaign_id = str(uuid.uuid4())
    campaign_dir = DATA_DIR / "campaigns" / f"{brand_id}_{campaign_id}"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    campaign_data = {
        "id": campaign_id,
        "brand_id": brand_id,
        "brand_name": brand.get("name"),
        "adn_version": campaign.adn_version or adn.get("version", "draft"),
        "name": campaign.name,
        "objective": campaign.objective,
        "secondary_objective": campaign.secondary_objective,
        "product_or_topic": campaign.product_or_topic,
        "target_audience": campaign.target_audience,
        "start_date": campaign.start_date,
        "end_date": campaign.end_date,
        "channels": campaign.channels,
        "frequency": campaign.frequency,
        "budget": campaign.budget,
        "restrictions": campaign.restrictions,
        "status": "generating",   # generating | active | paused | completed
        "publications_count": 0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    save_json(campaign_dir / "campaign.json", campaign_data)

    background_tasks.add_task(_generate_campaign_plan, brand_id, campaign_id, campaign_data, adn)
    return campaign_data


async def _generate_campaign_plan(brand_id: str, campaign_id: str,
                                   campaign_data: dict, adn: dict):
    """Genera la planificación temporal y publicaciones de la campaña.

    Estrategia de generación por lotes:
    1. Paso 1: Generar la estructura de etapas (stages) y el calendario base.
    2. Paso 2: Para cada etapa, generar las publicaciones en lotes pequeños
               (máximo MAX_PUBS_PER_BATCH por llamada al LLM).

    Esto evita que el LLM se sature con prompts muy largos y produzca
    publicaciones incompletas o truncadas.
    """
    import time
    MAX_PUBS_PER_BATCH = 5  # Máximo de publicaciones por llamada al LLM
    start = time.time()
    campaign_dir = DATA_DIR / "campaigns" / f"{brand_id}_{campaign_id}"

    try:
        model = get_active_model()
        adn_summary = json.dumps(adn.get("fields", {}), ensure_ascii=False)[:2000]
        channels_str = ', '.join(campaign_data['channels'])
        system_prompt = get_system_prompt("campaign_strategist") or _get_campaign_strategist_prompt()

        # ── Paso 1: Generar estructura de etapas y calendario ────────────────
        logger.info(f"[Campaña {campaign_id}] Paso 1: generando estructura de etapas...")
        stages_message = (
            f"Crea la estructura de etapas para esta campaña de marketing.\n\n"
            f"CAMPAÑA: {campaign_data['name']}\n"
            f"OBJETIVO: {campaign_data['objective']}\n"
            f"PRODUCTO/TEMA: {campaign_data['product_or_topic']}\n"
            f"AUDIENCIA: {campaign_data['target_audience']}\n"
            f"PERÍODO: {campaign_data['start_date']} al {campaign_data['end_date']}\n"
            f"CANALES: {channels_str}\n"
            f"FRECUENCIA: {campaign_data['frequency']}\n\n"
            f"ADN DE MARCA:\n{adn_summary}\n\n"
            f"INSTRUCCIONES:\n"
            f"- Responde SOLO con JSON válido, sin texto antes ni después.\n"
            f"- Define entre 3 y 5 etapas narrativas para la campaña.\n"
            f"- Para cada etapa, indica: name, description, days (rango), focus, channels_priority.\n"
            f"- Calcula cuántas publicaciones hay por etapa según la frecuencia y los canales.\n\n"
            f"Formato JSON:\n"
            f'{{"stages": [{{"name": "Descubrimiento", "description": "...", "days": "1-5", '
            f'"focus": "awareness", "publications_count": 3}}]}}'
        )

        stages_result = call_ollama(
            model, system_prompt, stages_message,
            temperature=0.4,
            timeout=get_ollama_timeout("adn"),
        )
        stages_parsed = _extract_json_from_llm(stages_result)
        stages = (stages_parsed or {}).get("stages") or [
            {"name": "Descubrimiento", "description": "Presentación y awareness",        "days": "1-3",  "focus": "awareness"},
            {"name": "Consideración",  "description": "Beneficios y propuesta de valor",  "days": "4-8",  "focus": "engagement"},
            {"name": "Activación",     "description": "CTA directo y conversión",         "days": "9-12", "focus": "conversion"},
            {"name": "Cierre",         "description": "Urgencia y recordación",           "days": "13+",  "focus": "retention"},
        ]
        logger.info(f"[Campaña {campaign_id}] Etapas generadas: {[s['name'] for s in stages]}")

        # ── Paso 2: Calcular calendario de publicaciones ─────────────────────
        start_dt  = datetime.strptime(campaign_data["start_date"], "%Y-%m-%d")
        end_dt    = datetime.strptime(campaign_data["end_date"],   "%Y-%m-%d")
        total_days = (end_dt - start_dt).days + 1
        channels   = campaign_data.get("channels", ["Instagram"])
        frequency  = campaign_data.get("frequency", "diaria")

        # Determinar frecuencia en días
        freq_map = {
            "diaria": 1, "daily": 1,
            "cada 2 dias": 2, "cada 2 días": 2, "every 2 days": 2,
            "semanal": 7, "weekly": 7,
            "bisemanal": 4, "twice a week": 4,
        }
        freq_days = freq_map.get(frequency.lower(), 1)

        # Construir lista de slots (fecha, canal, etapa)
        slots = []
        for day_offset in range(0, total_days, freq_days):
            current_date = start_dt + timedelta(days=day_offset)
            stage_idx = min(int(day_offset / max(total_days / len(stages), 1)), len(stages) - 1)
            stage = stages[stage_idx]
            for channel in channels:
                slots.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "channel": channel,
                    "stage": stage["name"],
                    "stage_focus": stage.get("focus", "awareness"),
                })

        logger.info(f"[Campaña {campaign_id}] Total slots calculados: {len(slots)} publicaciones")

        # ── Paso 3: Generar publicaciones en lotes ───────────────────────────
        all_publications = []
        total_batches = (len(slots) + MAX_PUBS_PER_BATCH - 1) // MAX_PUBS_PER_BATCH

        for batch_idx in range(total_batches):
            batch_slots = slots[batch_idx * MAX_PUBS_PER_BATCH : (batch_idx + 1) * MAX_PUBS_PER_BATCH]
            logger.info(
                f"[Campaña {campaign_id}] Lote {batch_idx + 1}/{total_batches}: "
                f"{len(batch_slots)} publicaciones..."
            )

            # Actualizar progreso en el archivo de campaña (por canal)
            try:
                camp_file = campaign_dir / "campaign.json"
                camp_progress = load_json(camp_file)
                camp_progress["status"] = "generating"

                # Calcular totales por canal
                channel_totals = {}
                for s in slots:
                    ch = s["channel"]
                    channel_totals[ch] = channel_totals.get(ch, 0) + 1

                # Calcular generados por canal hasta ahora
                channel_done = {}
                for p in all_publications:
                    ch = p.get("channel", "")
                    channel_done[ch] = channel_done.get(ch, 0) + 1

                channel_progress = []
                for ch, total in channel_totals.items():
                    channel_progress.append({
                        "channel": ch,
                        "done": channel_done.get(ch, 0),
                        "total": total,
                        "pct": int(channel_done.get(ch, 0) * 100 / total) if total > 0 else 0,
                    })

                camp_progress["generation_progress"] = {
                    "batch": batch_idx + 1,
                    "total_batches": total_batches,
                    "publications_done": len(all_publications),
                    "publications_total": len(slots),
                    "pct": int(len(all_publications) * 100 / len(slots)) if slots else 0,
                    "channels": channel_progress,
                }
                camp_progress["updated_at"] = datetime.utcnow().isoformat()
                save_json(camp_file, camp_progress)
            except Exception:
                pass

            # Construir prompt del lote
            slots_desc = "\n".join([
                f"  {i+1}. Canal: {s['channel']}, Fecha: {s['date']} 10:00, "
                f"Etapa: {s['stage']} (foco: {s['stage_focus']})"
                for i, s in enumerate(batch_slots)
            ])

            batch_message = (
                f"Genera exactamente {len(batch_slots)} publicaciones de marketing para las siguientes posiciones.\n\n"
                f"DATOS DE LA CAMPAÑA:\n"
                f"- Producto/Tema: {campaign_data['product_or_topic']}\n"
                f"- Audiencia: {campaign_data['target_audience']}\n"
                f"- Objetivo: {campaign_data['objective']}\n"
                f"- Marca: {campaign_data.get('brand_name', '')}\n\n"
                f"ADN DE MARCA (resumen):\n{adn_summary[:1000]}\n\n"
                f"POSICIONES A GENERAR:\n{slots_desc}\n\n"
                f"INSTRUCCIONES IMPORTANTES:\n"
                f"- Responde SOLO con JSON válido, sin texto antes ni después.\n"
                f"- Genera EXACTAMENTE {len(batch_slots)} publicaciones en el array 'publications'.\n"
                f"- El campo 'text' debe ser el TEXTO REAL del post (no JSON, no descripción).\n"
                f"- El texto debe ser persuasivo, natural y adaptado al canal y etapa.\n"
                f"- Usa acentos y caracteres especiales correctamente (español).\n"
                f"- Cada publicación debe tener: channel, scheduled_at, stage, objective, "
                f"text, hashtags (array), cta, image_prompt.\n\n"
                f"Formato JSON:\n"
                f'{{"publications": [{{"channel": "...", "scheduled_at": "YYYY-MM-DD HH:MM", '
                f'"stage": "...", "objective": "...", "text": "texto real del post", '
                f'"hashtags": ["#tag1"], "cta": "...", "image_prompt": "..."}}]}}'
            )

            try:
                batch_result = call_ollama(
                    model, system_prompt, batch_message,
                    temperature=0.6,
                    timeout=get_ollama_timeout("campaign"),
                )
                batch_parsed = _extract_json_from_llm(batch_result)
                batch_pubs = (batch_parsed or {}).get("publications", [])

                if batch_pubs:
                    # Asignar IDs y campos requeridos
                    for i, pub in enumerate(batch_pubs):
                        pub["id"] = str(uuid.uuid4())
                        pub["campaign_id"] = campaign_data["id"]
                        pub["brand_id"] = campaign_data["brand_id"]
                        pub.setdefault("status", "pending")
                        pub.setdefault("edit_status", "draft")

                        # Sanear texto si contiene JSON crudo
                        raw_text = pub.get("text", "")
                        if not raw_text or raw_text.strip().startswith("{") or \
                           "\"stages\"" in raw_text or "\"publications\"" in raw_text:
                            slot = batch_slots[i] if i < len(batch_slots) else batch_slots[-1]
                            pub["text"] = _build_fallback_post_text(
                                slot["channel"], slot["stage"], campaign_data
                            )
                            pub["edit_status"] = "needs_review"

                        # Asegurar scheduled_at del slot si el LLM lo omitió
                        if not pub.get("scheduled_at") and i < len(batch_slots):
                            pub["scheduled_at"] = batch_slots[i]["date"] + " 10:00"
                        if not pub.get("channel") and i < len(batch_slots):
                            pub["channel"] = batch_slots[i]["channel"]
                        if not pub.get("stage") and i < len(batch_slots):
                            pub["stage"] = batch_slots[i]["stage"]

                    all_publications.extend(batch_pubs)
                    logger.info(f"[Campaña {campaign_id}] Lote {batch_idx+1}: {len(batch_pubs)} pubs OK")
                else:
                    # Fallback: generar publicaciones básicas para este lote
                    logger.warning(f"[Campaña {campaign_id}] Lote {batch_idx+1}: LLM no retornó publicaciones, usando fallback")
                    for slot in batch_slots:
                        all_publications.append({
                            "id": str(uuid.uuid4()),
                            "campaign_id": campaign_data["id"],
                            "brand_id": campaign_data["brand_id"],
                            "channel": slot["channel"],
                            "scheduled_at": slot["date"] + " 10:00",
                            "stage": slot["stage"],
                            "objective": campaign_data["objective"],
                            "text": _build_fallback_post_text(slot["channel"], slot["stage"], campaign_data),
                            "hashtags": ["#marca", "#marketing"],
                            "cta": "¡Contáctanos!",
                            "image_prompt": f"Imagen para {slot['channel']} sobre {campaign_data['product_or_topic']}",
                            "status": "pending",
                            "edit_status": "needs_review",
                        })

            except Exception as batch_err:
                logger.error(f"[Campaña {campaign_id}] Error en lote {batch_idx+1}: {batch_err}")
                # Generar fallback para este lote y continuar
                for slot in batch_slots:
                    all_publications.append({
                        "id": str(uuid.uuid4()),
                        "campaign_id": campaign_data["id"],
                        "brand_id": campaign_data["brand_id"],
                        "channel": slot["channel"],
                        "scheduled_at": slot["date"] + " 10:00",
                        "stage": slot["stage"],
                        "objective": campaign_data["objective"],
                        "text": _build_fallback_post_text(slot["channel"], slot["stage"], campaign_data),
                        "hashtags": ["#marca", "#marketing"],
                        "cta": "¡Contáctanos!",
                        "image_prompt": f"Imagen para {slot['channel']} sobre {campaign_data['product_or_topic']}",
                        "status": "pending",
                        "edit_status": "needs_review",
                    })

        # ── Guardar plan completo ────────────────────────────────────────────
        plan = {"stages": stages, "publications": all_publications}
        save_json(campaign_dir / "plan.json", plan)

        # Actualizar estado de la campaña
        camp_file = campaign_dir / "campaign.json"
        camp = load_json(camp_file)
        camp["status"] = "active"
        camp["publications_count"] = len(all_publications)
        camp["stages_count"] = len(stages)
        camp["updated_at"] = datetime.utcnow().isoformat()
        camp.pop("generation_progress", None)  # Limpiar progreso
        save_json(camp_file, camp)

        latency = int((time.time() - start) * 1000)
        log_audit("campaign_strategist", "generate_campaign_plan",
                  {"brand_id": brand_id, "campaign_id": campaign_id},
                  f"{len(all_publications)} publicaciones en {total_batches} lotes",
                  model, latency, True)
        logger.info(
            f"[Campaña {campaign_id}] Generación completa: "
            f"{len(all_publications)} publicaciones en {total_batches} lotes, "
            f"{latency/1000:.1f}s total"
        )

    except Exception as e:
        camp_file = campaign_dir / "campaign.json"
        camp = load_json(camp_file)
        camp["status"] = "error"
        camp["error"] = str(e)
        camp["updated_at"] = datetime.utcnow().isoformat()
        save_json(camp_file, camp)
        log_audit("campaign_strategist", "generate_campaign_plan",
                  {"brand_id": brand_id, "campaign_id": campaign_id},
                  "", get_active_model(), 0, False, str(e))
        logger.error(f"Error generando campaña {campaign_id}: {e}")


def _extract_json_from_llm(text: str) -> Optional[dict]:
    """Extrae el primer objeto JSON válido de la respuesta del LLM.

    Maneja los casos más comunes en Windows/Ollama antiguo:
    - JSON envuelto en bloques ```json ... ```
    - JSON precedido de texto explicativo
    - JSON con objetos anidados de profundidad variable
    """
    import re

    # 1. Eliminar bloques de código markdown
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # 2. Intentar parsear el texto completo directamente
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Buscar el primer '{' y encontrar el JSON balanceado desde ahí
    start_idx = cleaned.find('{')
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end_idx = -1

    for i, ch in enumerate(cleaned[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx == -1:
        return None

    candidate = cleaned[start_idx:end_idx + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 4. Último recurso: regex greedy (comportamiento anterior)
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _parse_campaign_plan(llm_output: str, campaign_data: dict) -> dict:
    """Parsea el plan de campaña del LLM, con fallback a estructura básica.

    Usa _extract_json_from_llm para manejar correctamente respuestas con
    bloques markdown, texto previo o JSON parcialmente malformado.
    """
    parsed = _extract_json_from_llm(llm_output)

    if parsed and "publications" in parsed:
        # Asegurar que cada publicación tenga ID y campos requeridos
        for pub in parsed["publications"]:
            if "id" not in pub:
                pub["id"] = str(uuid.uuid4())
            pub.setdefault("status", "pending")
            pub.setdefault("edit_status", "draft")
            pub.setdefault("campaign_id", campaign_data["id"])
            pub.setdefault("brand_id", campaign_data["brand_id"])

            # Sanear el campo text: si contiene JSON crudo, reemplazarlo
            raw_text = pub.get("text", "")
            if raw_text and (raw_text.strip().startswith("{") or "\"stages\"" in raw_text or "\"publications\"" in raw_text):
                pub["text"] = _build_fallback_post_text(
                    pub.get("channel", "Red social"),
                    pub.get("stage", ""),
                    campaign_data,
                )
                pub["edit_status"] = "needs_review"

        return parsed

    # Fallback completo: generar publicaciones básicas con texto limpio
    start_dt = datetime.strptime(campaign_data["start_date"], "%Y-%m-%d")
    end_dt   = datetime.strptime(campaign_data["end_date"],   "%Y-%m-%d")
    days     = (end_dt - start_dt).days + 1
    channels = campaign_data.get("channels", ["Instagram"])

    stages = [
        {"name": "Descubrimiento", "description": "Presentación y awareness",        "days": "1-3"},
        {"name": "Consideración",  "description": "Beneficios y propuesta de valor",  "days": "4-8"},
        {"name": "Activación",     "description": "CTA directo y conversión",         "days": "9-12"},
        {"name": "Cierre",         "description": "Urgencia y recordación",           "days": "13+"},
    ]

    publications = []
    pub_count = 0
    for day_offset in range(min(days, 15)):
        current_date = start_dt + timedelta(days=day_offset)
        stage_idx    = min(day_offset // 4, len(stages) - 1)

        for channel in channels[:2]:  # máximo 2 canales en fallback
            pub_count += 1
            publications.append({
                "id":           str(uuid.uuid4()),
                "campaign_id":  campaign_data["id"],
                "brand_id":     campaign_data["brand_id"],
                "channel":      channel,
                "scheduled_at": current_date.strftime("%Y-%m-%d") + " 10:00",
                "stage":        stages[stage_idx]["name"],
                "objective":    campaign_data["objective"],
                # Texto limpio — nunca incluir el output crudo del LLM
                "text":         _build_fallback_post_text(channel, stages[stage_idx]["name"], campaign_data),
                "hashtags":     ["#marca", "#marketing", "#pyme"],
                "cta":          "¡Contáctanos!",
                "image_prompt": f"Imagen para {channel} sobre {campaign_data['product_or_topic']}",
                "status":       "pending",
                "edit_status":  "needs_review",  # indica que necesita revisión manual
            })

    logger.warning(
        f"No se pudo parsear el plan del LLM para campaña {campaign_data['id']}. "
        f"Usando fallback con {len(publications)} publicaciones básicas."
    )
    return {"stages": stages, "publications": publications, "raw_plan": llm_output[:2000]}


def _build_fallback_post_text(channel: str, stage: str, campaign_data: dict) -> str:
    """Genera un texto de post genérico pero limpio cuando el LLM no devuelve texto válido."""
    product = campaign_data.get("product_or_topic", "nuestros servicios")
    audience = campaign_data.get("target_audience", "nuestros clientes")
    brand = campaign_data.get("brand_name", "")

    templates = {
        "Descubrimiento": (
            f"¿Conoces {product}? "
            f"{'En ' + brand + ', te' if brand else 'Te'} presentamos una solución pensada para {audience}. "
            f"¡Sigue nuestra cuenta para saber más!"
        ),
        "Consideración": (
            f"{product}: la herramienta que {audience} necesita. "
            f"Descubrí cómo podemos ayudarte a alcanzar tus objetivos. "
            f"¡Contáctanos hoy!"
        ),
        "Activación": (
            f"¡Es el momento de actuar! "
            f"{product} está disponible para {audience}. "
            f"No dejes pasar esta oportunidad. ¡Escribinos ahora!"
        ),
        "Cierre": (
            f"Últimos días para aprovechar nuestra propuesta en {product}. "
            f"{audience}: ¡no te quedes sin tu lugar! ¡Reservá hoy!"
        ),
    }
    base_text = templates.get(stage, templates["Descubrimiento"])

    # Agregar indicación del canal si es relevante
    channel_hint = {
        "LinkedIn":  " #networking #profesionales",
        "Instagram": " ✨ #emprendimiento",
        "Facebook":  " 📌 ¡Compartilo con tu red!",
        "Twitter":   " #pyme",
        "TikTok":    " 🎥 ¡Mirá nuestro video!",
    }.get(channel, "")

    return base_text + channel_hint


def _get_campaign_strategist_prompt() -> str:
    return """Eres un estratega de marketing digital experto en campañas para PYMEs.
Tu tarea es crear una planificación temporal detallada para una campaña de marketing.

REGLAS:
- Organiza el contenido en etapas narrativas coherentes
- Adapta el tono y CTA según la etapa (descubrimiento → consideración → activación → cierre)
- Respeta el ADN de marca en cada pieza
- Genera publicaciones específicas para cada canal con sus convenciones

FORMATO DE RESPUESTA (JSON obligatorio):
{
  "stages": [
    {"name": "Nombre etapa", "description": "Descripción", "days": "1-3", "focus": "objetivo"}
  ],
  "publications": [
    {
      "channel": "Instagram",
      "scheduled_at": "2024-01-15 10:00",
      "stage": "Descubrimiento",
      "objective": "Awareness",
      "text": "Texto del post...",
      "hashtags": ["#hashtag1", "#hashtag2"],
      "cta": "Llamada a la acción",
      "image_prompt": "Descripción de imagen para generar",
      "justification": "Por qué esta pieza en este momento"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# RUTAS: Publicaciones
# ---------------------------------------------------------------------------
@app.get("/api/campaigns/{campaign_id}/publications")
def get_publications(campaign_id: str, channel: Optional[str] = None,
                     status: Optional[str] = None):
    """Lista las publicaciones de una campaña con filtros opcionales."""
    # Buscar el directorio de la campaña
    for camp_dir in (DATA_DIR / "campaigns").iterdir():
        if camp_dir.is_dir() and campaign_id in camp_dir.name:
            plan = load_json(camp_dir / "plan.json", {"publications": []})
            publications = plan.get("publications", [])

            if channel:
                publications = [p for p in publications if p.get("channel") == channel]
            if status:
                publications = [p for p in publications if p.get("status") == status]

            return {"publications": publications, "total": len(publications)}

    raise HTTPException(status_code=404, detail="Campaña no encontrada")


@app.get("/api/campaigns/{campaign_id}/publications/{pub_id}")
def get_publication(campaign_id: str, pub_id: str):
    """Retorna el detalle de una publicación específica."""
    for camp_dir in (DATA_DIR / "campaigns").iterdir():
        if camp_dir.is_dir() and campaign_id in camp_dir.name:
            plan = load_json(camp_dir / "plan.json", {"publications": []})
            for pub in plan.get("publications", []):
                if pub.get("id") == pub_id:
                    return pub
    raise HTTPException(status_code=404, detail="Publicación no encontrada")


@app.put("/api/campaigns/{campaign_id}/publications/{pub_id}")
def update_publication(campaign_id: str, pub_id: str, update: PublicationUpdate):
    """Actualiza una publicación (texto, hashtags, estado, etc.)."""
    for camp_dir in (DATA_DIR / "campaigns").iterdir():
        if camp_dir.is_dir() and campaign_id in camp_dir.name:
            plan_file = camp_dir / "plan.json"
            plan = load_json(plan_file, {"publications": []})

            for pub in plan.get("publications", []):
                if pub.get("id") == pub_id:
                    update_data = update.dict(exclude_none=True)
                    pub.update(update_data)
                    pub["updated_at"] = datetime.utcnow().isoformat()

                    # Registrar cambio de estado
                    if "status" in update_data:
                        pub["status_history"] = pub.get("status_history", [])
                        pub["status_history"].append({
                            "status": update_data["status"],
                            "timestamp": datetime.utcnow().isoformat(),
                        })

                    save_json(plan_file, plan)
                    return pub

    raise HTTPException(status_code=404, detail="Publicación no encontrada")


@app.post("/api/campaigns/{campaign_id}/publications/{pub_id}/regenerate")
async def regenerate_publication(campaign_id: str, pub_id: str,
                                  instruction: Optional[str] = None,
                                  language: Optional[str] = None,
                                  model: Optional[str] = None):
    """Regenera una publicación con instrucción, idioma y modelo opcionales.
    
    El parámetro `model` permite al frontend especificar qué modelo de texto
    usar para la regeneración, sin cambiar el modelo activo global.
    """
    import time, re
    start = time.time()

    lang_label = {
        "es": "español", "en": "English", "pt": "português"
    }.get(language or "es", "español")

    for camp_dir in (DATA_DIR / "campaigns").iterdir():
        if camp_dir.is_dir() and campaign_id in camp_dir.name:
            plan_file = camp_dir / "plan.json"
            plan = load_json(plan_file, {"publications": []})
            camp = load_json(camp_dir / "campaign.json", {})

            for pub in plan.get("publications", []):
                if pub.get("id") == pub_id:
                    brand_id = camp.get("brand_id")
                    adn = load_json(DATA_DIR / "brands" / brand_id / "adn.json") or \
                          load_json(DATA_DIR / "brands" / brand_id / "adn_draft.json", {})
                    adn_summary = json.dumps(adn.get("fields", {}), ensure_ascii=False)[:1500]

                    # Usar el modelo especificado o el activo global
                    model = model or get_active_model()
                    system_prompt = get_system_prompt("content_writer") or _get_content_writer_prompt()

                    user_message = (
                        f"Regenera esta publicación para {pub.get('channel')}:\n\n"
                        f"PUBLICACIÓN ACTUAL:\n{pub.get('text', '')}\n\n"
                        f"ETAPA: {pub.get('stage', '')}\n"
                        f"OBJETIVO: {pub.get('objective', '')}\n"
                        f"INSTRUCCIÓN: {instruction or 'Mejora la publicación manteniendo el ADN de marca'}\n"
                        f"IDIOMA DE SALIDA: {lang_label}\n\n"
                        f"ADN DE MARCA:\n{adn_summary}\n\n"
                        f"IMPORTANTE: Responde SOLO con JSON válido, sin bloques de código markdown, "
                        f"sin texto adicional antes ni después del JSON.\n"
                        f'Formato: {{"texto_del_post": "...", "hashtags": ["#tag1"], "cta": "...", "image_prompt": "..."}}'
                    )

                    result = call_ollama(
                        model, system_prompt, user_message,
                        temperature=0.8,
                        timeout=get_ollama_timeout("default"),
                    )
                    latency = int((time.time() - start) * 1000)

                    # Guardar versión anterior
                    pub["previous_versions"] = pub.get("previous_versions", [])
                    pub["previous_versions"].append({
                        "text": pub.get("text"),
                        "hashtags": pub.get("hashtags"),
                        "regenerated_at": datetime.utcnow().isoformat(),
                    })

                    # Parsear JSON limpio (elimina bloques ```json ... ``` si existen)
                    parsed = _parse_llm_json(result)
                    if parsed:
                        # Soportar tanto 'text' como 'texto_del_post'
                        pub["text"] = parsed.get("text") or parsed.get("texto_del_post") or pub.get("text", "")
                        pub["hashtags"] = parsed.get("hashtags", pub.get("hashtags", []))
                        pub["cta"] = parsed.get("cta", pub.get("cta", ""))
                        pub["image_prompt"] = parsed.get("image_prompt", pub.get("image_prompt", ""))
                    else:
                        # Si no hay JSON válido, usar el texto limpio directamente
                        pub["text"] = _strip_markdown_fences(result)

                    pub["edit_status"] = "regenerated"
                    pub["updated_at"] = datetime.utcnow().isoformat()
                    save_json(plan_file, plan)

                    log_audit("content_writer", "regenerate_publication",
                              {"campaign_id": campaign_id, "pub_id": pub_id},
                              result[:500], model, latency, True)
                    return pub

    raise HTTPException(status_code=404, detail="Publicación no encontrada")


class GenerateImageRequest(BaseModel):
    image_prompt: str
    model: str = "x/z-image-turbo"
    instruction: Optional[str] = None


def _ensure_model_available(model: str) -> dict:
    """
    Verifica si el modelo está disponible en Ollama.
    Si no lo está, lo descarga automáticamente (ollama pull).
    Retorna {"available": bool, "pulled": bool, "error": str|None}
    """
    try:
        # Listar modelos disponibles
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        resp.raise_for_status()
        tags = resp.json()
        available_models = [m["name"] for m in tags.get("models", [])]

        # Normalizar nombre: "x/z-image-turbo" puede aparecer como "z-image-turbo" o con tag
        model_base = model.split(":")[0]  # sin tag :latest
        model_found = any(
            m.startswith(model_base) or m.startswith(model)
            for m in available_models
        )

        if model_found:
            return {"available": True, "pulled": False, "error": None}

        # No está disponible — hacer pull
        logger.info(f"Modelo {model} no encontrado. Iniciando descarga...")
        pull_resp = requests.post(
            f"{OLLAMA_URL}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,  # 10 minutos para descargar
        )
        pull_resp.raise_for_status()
        pull_data = pull_resp.json()
        status = pull_data.get("status", "")
        if "success" in status.lower() or status == "":
            logger.info(f"Modelo {model} descargado correctamente")
            return {"available": True, "pulled": True, "error": None}
        else:
            return {"available": False, "pulled": False, "error": f"Pull status: {status}"}

    except requests.exceptions.ConnectionError:
        return {"available": False, "pulled": False, "error": "Ollama no está disponible"}
    except Exception as e:
        return {"available": False, "pulled": False, "error": str(e)}


def _get_image_provider_config() -> dict:
    """Retorna la configuración del proveedor de imágenes desde config.json o env vars."""
    cfg = load_json(DATA_DIR / "config.json", {})
    return {
        "provider": cfg.get("image_provider", IMAGE_PROVIDER),
        "a1111_url": cfg.get("a1111_url", A1111_URL),
        "comfyui_url": cfg.get("comfyui_url", COMFYUI_URL),
        "timeout": int(cfg.get("image_timeout", IMAGE_TIMEOUT)),
    }


def _try_automatic1111(prompt: str, cfg: dict) -> Optional[str]:
    """Intenta generar imagen con AUTOMATIC1111 WebUI.

    Requiere que AUTOMATIC1111 esté corriendo con --api flag:
      webui-user.bat: set COMMANDLINE_ARGS=--api
    URL por defecto: http://localhost:7860
    Endpoint: POST /sdapi/v1/txt2img
    Respuesta: {"images": ["base64..."]}
    """
    base_url = cfg.get("a1111_url", A1111_URL).rstrip("/")
    timeout = cfg.get("timeout", IMAGE_TIMEOUT)

    try:
        # Verificar que A1111 esté disponible
        health = requests.get(f"{base_url}/sdapi/v1/sd-models", timeout=5)
        if health.status_code != 200:
            logger.debug(f"A1111 no disponible en {base_url} (status {health.status_code})")
            return None
    except Exception:
        logger.debug(f"A1111 no disponible en {base_url}")
        return None

    logger.info(f"Generando imagen con AUTOMATIC1111 en {base_url}...")
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted, watermark, text",
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "cfg_scale": 7,
        "sampler_name": "DPM++ 2M Karras",
    }
    resp = requests.post(f"{base_url}/sdapi/v1/txt2img", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    images = data.get("images", [])
    if images and images[0]:
        logger.info(f"A1111: imagen generada, longitud base64: {len(images[0])}")
        return images[0]
    logger.warning("A1111: respuesta sin campo 'images'")
    return None


def _try_comfyui(prompt: str, cfg: dict) -> Optional[str]:
    """Intenta generar imagen con ComfyUI.

    Requiere que ComfyUI esté corriendo.
    URL por defecto: http://localhost:8188
    Usa el workflow mínimo de txt2img con SDXL-Turbo o SD 1.5.
    """
    import uuid as _uuid
    import time as _time

    base_url = cfg.get("comfyui_url", COMFYUI_URL).rstrip("/")
    timeout = cfg.get("timeout", IMAGE_TIMEOUT)

    try:
        health = requests.get(f"{base_url}/system_stats", timeout=5)
        if health.status_code != 200:
            logger.debug(f"ComfyUI no disponible en {base_url}")
            return None
    except Exception:
        logger.debug(f"ComfyUI no disponible en {base_url}")
        return None

    logger.info(f"Generando imagen con ComfyUI en {base_url}...")

    # Workflow mínimo para txt2img
    client_id = str(_uuid.uuid4())
    workflow = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 20, "cfg": 7,
            "sampler_name": "euler", "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0],
            "negative": ["7", 0], "latent_image": ["5", 0]
        }},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.ckpt"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "css_brand", "images": ["8", 0]}},
    }

    queue_resp = requests.post(
        f"{base_url}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    queue_resp.raise_for_status()
    prompt_id = queue_resp.json().get("prompt_id")
    if not prompt_id:
        logger.warning("ComfyUI: no se obtuvo prompt_id")
        return None

    # Esperar a que termine (polling)
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        hist_resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
        if hist_resp.status_code == 200:
            hist = hist_resp.json()
            if prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                for node_id, node_out in outputs.items():
                    images_list = node_out.get("images", [])
                    if images_list:
                        img_info = images_list[0]
                        # Descargar la imagen generada
                        img_resp = requests.get(
                            f"{base_url}/view",
                            params={"filename": img_info["filename"],
                                    "subfolder": img_info.get("subfolder", ""),
                                    "type": img_info.get("type", "output")},
                            timeout=30,
                        )
                        img_resp.raise_for_status()
                        import base64 as _b64
                        b64 = _b64.b64encode(img_resp.content).decode("utf-8")
                        logger.info(f"ComfyUI: imagen generada, longitud base64: {len(b64)}")
                        return b64
        _time.sleep(2)

    logger.warning("ComfyUI: timeout esperando resultado")
    return None


def _generate_placeholder_svg(prompt: str, model: str) -> str:
    """Genera un placeholder SVG profesional codificado en base64.

    Se usa como fallback cuando la generación de imágenes con Ollama no está
    disponible (ej: Windows). El SVG incluye el prompt como referencia visual
    para que el usuario sepa qué imagen debe colocar manualmente.
    """
    import base64 as _b64
    import html as _html

    # Truncar prompt para el SVG
    prompt_short = prompt[:120] + ("..." if len(prompt) > 120 else "")
    prompt_escaped = _html.escape(prompt_short)

    # Dividir el prompt en líneas de ~50 caracteres para el SVG
    words = prompt_escaped.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= 50:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    lines = lines[:5]  # máximo 5 líneas

    # Construir los tspan para las líneas del prompt
    tspans = ""
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else "1.4em"
        tspans += f'<tspan x="512" dy="{dy}">{line}</tspan>'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a1a2e;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#16213e;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="frame" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#e94560;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0f3460;stop-opacity:1" />
    </linearGradient>
  </defs>

  <!-- Fondo -->
  <rect width="1024" height="1024" fill="url(#bg)" />

  <!-- Marco decorativo -->
  <rect x="20" y="20" width="984" height="984" rx="16" ry="16"
        fill="none" stroke="url(#frame)" stroke-width="3" opacity="0.6" />
  <rect x="40" y="40" width="944" height="944" rx="12" ry="12"
        fill="none" stroke="#e94560" stroke-width="1" opacity="0.3" />

  <!-- Ícono de cámara / imagen -->
  <g transform="translate(512, 340)">
    <rect x="-80" y="-55" width="160" height="110" rx="12" ry="12"
          fill="none" stroke="#e94560" stroke-width="3" />
    <circle cx="0" cy="0" r="30" fill="none" stroke="#e94560" stroke-width="3" />
    <circle cx="0" cy="0" r="12" fill="#e94560" opacity="0.5" />
    <rect x="-15" y="-68" width="30" height="16" rx="4" ry="4"
          fill="none" stroke="#e94560" stroke-width="2" />
    <rect x="50" y="-52" width="18" height="12" rx="3" ry="3"
          fill="#e94560" opacity="0.4" />
  </g>

  <!-- Título -->
  <text x="512" y="480"
        font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="bold"
        fill="#e94560" text-anchor="middle" letter-spacing="3">
    IMAGEN PENDIENTE
  </text>

  <!-- Separador -->
  <line x1="200" y1="500" x2="824" y2="500" stroke="#e94560" stroke-width="1" opacity="0.4" />

  <!-- Prompt -->
  <text x="512" y="535"
        font-family="Arial, Helvetica, sans-serif" font-size="16"
        fill="#a0aec0" text-anchor="middle" dominant-baseline="hanging">
    {tspans}
  </text>

  <!-- Nota inferior -->
  <text x="512" y="920"
        font-family="Arial, Helvetica, sans-serif" font-size="13"
        fill="#4a5568" text-anchor="middle">
    Generación de imágenes no disponible en Windows · Reemplazar manualmente
  </text>

  <!-- Modelo -->
  <text x="512" y="945"
        font-family="Arial, Helvetica, sans-serif" font-size="11"
        fill="#2d3748" text-anchor="middle">
    Modelo solicitado: {_html.escape(model)}
  </text>
</svg>"""

    return _b64.b64encode(svg.encode("utf-8")).decode("utf-8")


@app.post("/api/campaigns/{campaign_id}/publications/{pub_id}/generate-image")
async def generate_publication_image(campaign_id: str, pub_id: str, req: GenerateImageRequest):
    """Genera una imagen para la publicación.

    Estrategia multi-proveedor (configurable via IMAGE_PROVIDER en config.json o env var):
    - "auto" (defecto): prueba en orden Ollama → A1111 → ComfyUI → placeholder SVG
    - "ollama": solo Ollama (macOS/Linux, experimental)
    - "automatic1111": AUTOMATIC1111 WebUI con --api flag
    - "comfyui": ComfyUI local

    En Windows, Ollama no soporta generación de imágenes. Si AUTOMATIC1111 o ComfyUI
    están instalados, se usan automáticamente. Si ninguno está disponible, se genera
    un placeholder SVG profesional que el usuario puede reemplazar manualmente.
    """
    import time, base64
    start = time.time()

    # Construir prompt completo
    prompt = req.image_prompt
    if req.instruction:
        prompt = f"{prompt}. Estilo adicional: {req.instruction}"

    logger.info(f"Generando imagen, proveedor configurado: {IMAGE_PROVIDER}, prompt: {prompt[:100]}...")

    image_b64: Optional[str] = None
    generation_method: str = "none"

    # Obtener configuración del proveedor (puede venir de config.json)
    img_cfg = _get_image_provider_config()
    provider = img_cfg["provider"]

    # -----------------------------------------------------------------------
    # Intento 0: Motor embebido (HuggingFace Diffusers + LCM)
    # Funciona en Windows, macOS y Linux sin dependencias externas.
    # El modelo se descarga automáticamente la primera vez (~2 GB).
    # -----------------------------------------------------------------------
    diffusion_model = img_cfg.get("diffusion_model", DEFAULT_DIFFUSION_MODEL)
    diffusion_steps = int(img_cfg.get("diffusion_steps", 4))

    if IMAGE_ENGINE_AVAILABLE and provider in ("auto", "embedded", "diffusers") and not image_b64:
        try:
            logger.info(f"[ImageEngine] Intentando motor embebido con modelo {diffusion_model}")
            engine_result = _engine_generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, distorted, ugly, watermark, text, logo",
                model_id=diffusion_model,
                steps=diffusion_steps,
                width=512,
                height=512,
                guidance_scale=1.0,
            )
            if engine_result.get("success") and engine_result.get("image_b64"):
                image_b64 = engine_result["image_b64"]
                generation_method = f"embedded_diffusers:{diffusion_model.split('/')[-1]}"
                logger.info(
                    f"[ImageEngine] Imagen generada en {engine_result.get('generation_time_s', '?')}s "
                    f"con {diffusion_model}"
                )
            else:
                logger.warning(f"[ImageEngine] Motor embebido falló: {engine_result.get('error')}. Probando siguiente.")
        except Exception as e:
            logger.warning(f"[ImageEngine] Error en motor embebido: {e}. Probando siguiente proveedor.")

    # -----------------------------------------------------------------------
    # Intento 1: Ollama (macOS/Linux con modelo de imagen)
    # -----------------------------------------------------------------------
    if provider in ("auto", "ollama") and not image_b64:
        try:
            model_status = _ensure_model_available(req.model)
            if model_status["available"]:
                if model_status.get("pulled"):
                    logger.info(f"Modelo {req.model} descargado antes de generar")

                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": req.model,
                        "prompt": prompt,
                        "stream": False,
                        "width": 1024,
                        "height": 1024,
                    },
                    timeout=img_cfg["timeout"],
                )

                if response.status_code == 200:
                    data = response.json()
                    if "image" in data and data["image"]:
                        image_b64 = data["image"]
                        generation_method = "ollama"
                        logger.info(f"Ollama: imagen generada, longitud: {len(image_b64)}")
                    elif "images" in data and data["images"]:
                        image_b64 = data["images"][0]
                        generation_method = "ollama"
                    elif "response" in data and data["response"]:
                        import re as _re
                        candidate = data["response"].strip().replace('\n', '').replace(' ', '')
                        if bool(_re.match(r'^[A-Za-z0-9+/=]{500,}$', candidate)):
                            image_b64 = candidate
                            generation_method = "ollama"
                else:
                    logger.warning(
                        f"Ollama retornó {response.status_code} para modelo de imagen. "
                        f"Esperado en Windows. Probando siguiente proveedor."
                    )
            else:
                logger.warning(f"Modelo Ollama {req.model} no disponible. Probando siguiente proveedor.")
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama no disponible. Probando siguiente proveedor.")
        except Exception as e:
            logger.warning(f"Error Ollama imagen: {e}. Probando siguiente proveedor.")

    # -----------------------------------------------------------------------
    # Intento 2: AUTOMATIC1111 WebUI
    # -----------------------------------------------------------------------
    if provider in ("auto", "automatic1111") and not image_b64:
        try:
            result = _try_automatic1111(prompt, img_cfg)
            if result:
                image_b64 = result
                generation_method = "automatic1111"
        except Exception as e:
            logger.warning(f"Error A1111: {e}. Probando siguiente proveedor.")

    # -----------------------------------------------------------------------
    # Intento 3: ComfyUI
    # -----------------------------------------------------------------------
    if provider in ("auto", "comfyui") and not image_b64:
        try:
            result = _try_comfyui(prompt, img_cfg)
            if result:
                image_b64 = result
                generation_method = "comfyui"
        except Exception as e:
            logger.warning(f"Error ComfyUI: {e}. Probando siguiente proveedor.")

    # -----------------------------------------------------------------------
    # Intento 4: Placeholder SVG profesional (fallback universal)
    # -----------------------------------------------------------------------
    if not image_b64:
        logger.info(
            f"Ningún proveedor de imagen disponible. "
            f"Generando placeholder SVG para: {prompt[:80]}"
        )
        image_b64 = _generate_placeholder_svg(prompt, req.model)
        generation_method = "placeholder_svg"

    # -----------------------------------------------------------------------
    # Guardar imagen / SVG en disco y actualizar publicación
    # -----------------------------------------------------------------------
    try:
        img_dir = DATA_DIR / "exports" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        # Determinar extensión según el método
        ext = "svg" if generation_method == "placeholder_svg" else "png"
        img_filename = f"{pub_id}_{ts}.{ext}"
        img_path = img_dir / img_filename

        if generation_method == "placeholder_svg":
            # El SVG ya es texto, no base64
            svg_content = base64.b64decode(image_b64).decode("utf-8")
            img_path.write_text(svg_content, encoding="utf-8")
            # Guardar también como .png (alias) para compatibilidad
            fixed_path = img_dir / f"{pub_id}.svg"
            fixed_path.write_text(svg_content, encoding="utf-8")
            img_bytes_len = len(svg_content.encode("utf-8"))
        else:
            img_bytes = base64.b64decode(image_b64, validate=False)
            img_path.write_bytes(img_bytes)
            fixed_path = img_dir / f"{pub_id}.png"
            fixed_path.write_bytes(img_bytes)
            img_bytes_len = len(img_bytes)

        logger.info(f"Imagen guardada: {img_path} ({img_bytes_len} bytes), método: {generation_method}")

        # Actualizar publicación con URL de imagen
        image_url = f"/api/images/{pub_id}.{ext}?t={ts}"
        pub_updated = False
        campaigns_root = DATA_DIR / "campaigns"
        if campaigns_root.exists():
            for camp_dir in campaigns_root.iterdir():
                if not camp_dir.is_dir():
                    continue
                if campaign_id in camp_dir.name or camp_dir.name == campaign_id:
                    plan_file = camp_dir / "plan.json"
                    if not plan_file.exists():
                        continue
                    plan = load_json(plan_file, {"publications": []})
                    for pub in plan.get("publications", []):
                        if pub.get("id") == pub_id:
                            pub["generated_image_url"] = f"/api/images/{pub_id}.{ext}"
                            pub["image_generation_method"] = generation_method
                            pub["updated_at"] = datetime.utcnow().isoformat()
                            save_json(plan_file, plan)
                            pub_updated = True
                            break
                if pub_updated:
                    break

        if not pub_updated:
            logger.warning(f"No se encontró la publicación {pub_id} para actualizar imagen")

        latency = int((time.time() - start) * 1000)
        log_audit("image_generator", "generate_image",
                  {"campaign_id": campaign_id, "pub_id": pub_id, "model": req.model},
                  f"Image {generation_method}: {img_filename} ({img_bytes_len} bytes)",
                  req.model, latency, True)

        result = {
            "image_url": image_url,
            "image_filename": img_filename,
            "image_size_bytes": img_bytes_len,
            "image_b64": image_b64,
            "generation_method": generation_method,
            "success": True,
        }

        # Agregar aviso si es placeholder
        if generation_method == "placeholder_svg":
            result["warning"] = (
                "La generación de imágenes con IA no está disponible en Windows todavía "
                "(limitación de Ollama). Se generó un placeholder visual con el prompt. "
                "Podes reemplazarla manualmente desde la UI o usar la función de regenerar."
            )

        return result

    except Exception as e:
        logger.error(f"Error guardando imagen: {e}")
        return {"error": str(e), "success": False}


@app.get("/api/images/{filename}")
def serve_generated_image(filename: str):
    """Sirve imágenes generadas por IA con headers anti-caché.
    Soporta .png (imagen real) y .svg (placeholder cuando Ollama no soporta imágenes).
    """
    # Soportar filename con query string (ej: pub_id.png?t=123)
    clean_filename = filename.split("?")[0]
    img_dir = DATA_DIR / "exports" / "images"

    # Buscar el archivo exacto primero
    img_path = img_dir / clean_filename
    if not img_path.exists():
        # Si pidieron .png pero solo existe .svg (placeholder), servir el SVG
        if clean_filename.endswith(".png"):
            svg_path = img_dir / clean_filename.replace(".png", ".svg")
            if svg_path.exists():
                img_path = svg_path
                clean_filename = svg_path.name
            else:
                logger.warning(f"Imagen no encontrada: {img_path}")
                raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {clean_filename}")
        else:
            logger.warning(f"Imagen no encontrada: {img_path}")
            raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {clean_filename}")

    media_type = "image/svg+xml" if clean_filename.endswith(".svg") else "image/png"
    return FileResponse(
        str(img_path),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/images")
def list_generated_images():
    """Lista todas las imágenes generadas (para debug)."""
    img_dir = DATA_DIR / "exports" / "images"
    if not img_dir.exists():
        return {"images": []}
    images = [
        {
            "filename": f.name,
            "url": f"/api/images/{f.name}",
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in sorted(img_dir.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
    ]
    return {"images": images, "count": len(images)}


@app.get("/api/image-providers/status")
def get_image_providers_status():
    """Retorna el estado de disponibilidad de cada proveedor de imágenes.
    Permite al frontend mostrar qué proveedores están disponibles y cuál se usará.
    """
    img_cfg = _get_image_provider_config()
    provider = img_cfg["provider"]

    # Verificar motor embebido (Diffusers)
    engine_status = _engine_get_status() if IMAGE_ENGINE_AVAILABLE else {"state": "unavailable"}
    embedded_available = IMAGE_ENGINE_AVAILABLE  # Disponible si las dependencias están instaladas
    embedded_ready = IMAGE_ENGINE_AVAILABLE and engine_status.get("state") == "ready"

    # Verificar Ollama
    ollama_available = False
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_available = r.status_code == 200
    except Exception:
        pass

    # Verificar AUTOMATIC1111
    a1111_available = False
    try:
        r = requests.get(f"{img_cfg['a1111_url']}/sdapi/v1/sd-models", timeout=3)
        a1111_available = r.status_code == 200
    except Exception:
        pass

    # Verificar ComfyUI
    comfyui_available = False
    try:
        r = requests.get(f"{img_cfg['comfyui_url']}/system_stats", timeout=3)
        comfyui_available = r.status_code == 200
    except Exception:
        pass

    # Determinar proveedor efectivo (en orden de prioridad)
    if provider == "auto":
        if embedded_available:
            effective = "embedded_diffusers"
        elif ollama_available:
            effective = "ollama"
        elif a1111_available:
            effective = "automatic1111"
        elif comfyui_available:
            effective = "comfyui"
        else:
            effective = "placeholder_svg"
    else:
        effective = provider

    return {
        "configured_provider": provider,
        "effective_provider": effective,
        "providers": {
            "embedded_diffusers": {
                "available": embedded_available,
                "ready": embedded_ready,
                "url": None,
                "note": "Motor IA embebido (HuggingFace Diffusers + LCM). Funciona en Windows, macOS y Linux sin configuración adicional.",
                "engine_status": engine_status,
                "model": img_cfg.get("diffusion_model", DEFAULT_DIFFUSION_MODEL),
                "available_models": _engine_list_models() if IMAGE_ENGINE_AVAILABLE else [],
            },
            "ollama": {
                "available": ollama_available,
                "url": OLLAMA_URL,
                "note": "Solo disponible en macOS/Linux (limitación Ollama)",
            },
            "automatic1111": {
                "available": a1111_available,
                "url": img_cfg["a1111_url"],
                "note": "Requiere AUTOMATIC1111 corriendo con --api flag",
                "install_url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui",
            },
            "comfyui": {
                "available": comfyui_available,
                "url": img_cfg["comfyui_url"],
                "note": "Requiere ComfyUI corriendo",
                "install_url": "https://github.com/comfyanonymous/ComfyUI",
            },
            "placeholder_svg": {
                "available": True,
                "url": None,
                "note": "Siempre disponible. Genera un placeholder visual para reemplazar manualmente.",
            },
        },
    }


@app.get("/api/image-engine/status")
def get_image_engine_status():
    """Retorna el estado detallado del motor de imagen embebido."""
    if not IMAGE_ENGINE_AVAILABLE:
        return {
            "available": False,
            "state": "unavailable",
            "message": "Motor no disponible. Instala: pip install torch diffusers transformers accelerate safetensors",
            "models": [],
        }
    status = _engine_get_status()
    return {
        "available": True,
        "state": status["state"],
        "model": status.get("model"),
        "progress": status.get("progress", 0),
        "message": status.get("message"),
        "error": status.get("error"),
        "ready": status["state"] == "ready",
        "models": _engine_list_models(),
    }


@app.post("/api/image-engine/load")
def load_image_engine(body: dict = None):
    """Inicia la carga del motor de imagen en background.

    Acepta un body JSON opcional con:
    - model_id: ID del modelo HuggingFace a cargar (default: LCM_Dreamshaper_v7)
    """
    if not IMAGE_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Motor de imagen no disponible. Instala las dependencias.")

    model_id = (body or {}).get("model_id", DEFAULT_DIFFUSION_MODEL)
    current = _engine_get_status()

    if current["state"] == "loading":
        return {"message": f"Motor ya está cargando: {current['message']}", "status": current}
    if current["state"] == "ready" and current.get("model") == model_id:
        return {"message": "Motor ya está listo", "status": current}

    _engine_load_async(model_id)
    return {"message": f"Carga iniciada para {model_id}", "model": model_id}


@app.post("/api/image-engine/unload")
def unload_image_engine():
    """Descarga el motor de imagen de memoria RAM.
    Útil si se necesita liberar RAM para el LLM.
    """
    if not IMAGE_ENGINE_AVAILABLE:
        return {"message": "Motor no disponible"}
    _engine_unload()
    return {"message": "Motor descargado de memoria"}


@app.post("/api/campaigns/{campaign_id}/publications/{pub_id}/upload-image")
async def upload_publication_image(
    campaign_id: str,
    pub_id: str,
    file: UploadFile = File(...),
):
    """Permite al usuario cargar una imagen manualmente desde su disco.

    El archivo se guarda en el directorio de imágenes del plugin y se asocia
    a la publicación. Soporta PNG, JPG, JPEG, GIF, WEBP y SVG.
    Máximo 10 MB.
    """
    import time

    # Validar tipo de archivo
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/svg+xml"}
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {content_type}. Use PNG, JPG, GIF, WEBP o SVG."
        )

    # Leer contenido
    content = await file.read()

    # Validar tamaño (máx 10 MB)
    max_size = 10 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Archivo demasiado grande: {len(content) // 1024} KB. Máximo 10 MB."
        )

    # Determinar extensión
    ext_map = {
        "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
        "image/gif": "gif", "image/webp": "webp", "image/svg+xml": "svg",
    }
    ext = ext_map.get(content_type, "png")

    # Guardar archivo
    img_dir = DATA_DIR / "exports" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    img_filename = f"{pub_id}_{ts}.{ext}"
    img_path = img_dir / img_filename
    img_path.write_bytes(content)

    # Guardar también como nombre fijo
    fixed_path = img_dir / f"{pub_id}.{ext}"
    fixed_path.write_bytes(content)

    # Actualizar publicación
    image_url = f"/api/images/{pub_id}.{ext}?t={ts}"
    pub_updated = False
    campaigns_root = DATA_DIR / "campaigns"
    if campaigns_root.exists():
        for camp_dir in campaigns_root.iterdir():
            if not camp_dir.is_dir():
                continue
            if campaign_id in camp_dir.name or camp_dir.name == campaign_id:
                plan_file = camp_dir / "plan.json"
                if not plan_file.exists():
                    continue
                plan = load_json(plan_file, {"publications": []})
                for pub in plan.get("publications", []):
                    if pub.get("id") == pub_id:
                        pub["generated_image_url"] = f"/api/images/{pub_id}.{ext}"
                        pub["image_generation_method"] = "manual_upload"
                        pub["updated_at"] = datetime.utcnow().isoformat()
                        save_json(plan_file, plan)
                        pub_updated = True
                        break
            if pub_updated:
                break

    logger.info(f"Imagen subida manualmente: {img_filename} ({len(content)} bytes) para pub {pub_id}")
    log_audit("image_generator", "upload_image",
              {"campaign_id": campaign_id, "pub_id": pub_id, "filename": file.filename},
              f"Manual upload: {img_filename} ({len(content)} bytes)",
              "manual", 0, True)

    return {
        "image_url": image_url,
        "image_filename": img_filename,
        "image_size_bytes": len(content),
        "generation_method": "manual_upload",
        "success": True,
    }


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extrae y parsea JSON de la respuesta del LLM, eliminando bloques markdown."""
    import re
    # Eliminar bloques ```json ... ``` o ``` ... ```
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    cleaned = cleaned.strip()

    # Intentar parsear directamente
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Buscar el primer objeto JSON válido en el texto
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _strip_markdown_fences(text: str) -> str:
    """Elimina bloques de código markdown del texto."""
    import re
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    return cleaned.strip()


def _get_content_writer_prompt() -> str:
    return """Eres un redactor creativo especializado en marketing digital para PYMEs.
Tu tarea es crear o mejorar publicaciones para redes sociales respetando el ADN de marca.

REGLAS:
- Mantén el tono y personalidad de la marca
- Adapta el formato al canal (Instagram: visual+emocional, LinkedIn: profesional, etc.)
- Incluye CTA claro y hashtags relevantes
- Sé conciso pero impactante

FORMATO DE RESPUESTA (JSON):
{
  "text": "Texto principal del post",
  "hashtags": ["#hashtag1", "#hashtag2"],
  "cta": "Llamada a la acción",
  "image_prompt": "Descripción detallada de imagen a generar"
}"""


# ---------------------------------------------------------------------------
# RUTAS: Agentes y Configuración
# ---------------------------------------------------------------------------
@app.get("/api/stats")
def get_stats():
    """Retorna estadísticas globales del sistema: marcas, ADN, campañas y publicaciones."""
    brands_file = DATA_DIR / "brands.json"
    brands_data = load_json(brands_file, {"brands": []})
    brands = brands_data.get("brands", [])

    total_brands = len(brands)
    adn_complete = sum(1 for b in brands if b.get("onboarding_status") == "complete")

    # Contar campañas y publicaciones recorriendo los directorios
    total_campaigns = 0
    total_publications = 0
    pub_by_status = {"pending": 0, "ready": 0, "published": 0, "omitted": 0}
    pub_by_channel = {}
    recent_campaigns = []

    campaigns_root = DATA_DIR / "campaigns"
    if campaigns_root.exists():
        for camp_dir in sorted(campaigns_root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
            if not camp_dir.is_dir():
                continue
            camp_file = camp_dir / "campaign.json"
            plan_file = camp_dir / "plan.json"
            if not camp_file.exists():
                continue
            camp = load_json(camp_file, {})
            total_campaigns += 1
            if len(recent_campaigns) < 5:
                recent_campaigns.append({
                    "id": camp.get("id"),
                    "name": camp.get("name"),
                    "brand_id": camp.get("brand_id"),
                    "start_date": camp.get("start_date"),
                    "end_date": camp.get("end_date"),
                    "channels": camp.get("channels", []),
                })
            if plan_file.exists():
                plan = load_json(plan_file, {"publications": []})
                pubs = plan.get("publications", [])
                total_publications += len(pubs)
                for pub in pubs:
                    status = pub.get("status", "pending")
                    pub_by_status[status] = pub_by_status.get(status, 0) + 1
                    channel = pub.get("channel", "Otro")
                    pub_by_channel[channel] = pub_by_channel.get(channel, 0) + 1

    return {
        "brands": total_brands,
        "adn_complete": adn_complete,
        "campaigns": total_campaigns,
        "publications": total_publications,
        "pub_by_status": pub_by_status,
        "pub_by_channel": pub_by_channel,
        "recent_campaigns": recent_campaigns,
    }


@app.get("/api/agents")
def list_agents():
    """Lista todos los agentes con su system_prompt real (desde disco) y contenido de skills."""
    agents_file = DATA_DIR / "agents" / "agents.json"
    agents_data = load_json(agents_file, {"agents": []})

    # Enriquecer cada agente con el prompt real desde disco y el contenido del skill file
    for agent in agents_data.get("agents", []):
        agent_id = agent.get("id", "")

        # 1. Leer system_prompt real desde DATA_DIR/prompts/system/{id}.md
        prompt_file = DATA_DIR / "prompts" / "system" / f"{agent_id}.md"
        if prompt_file.exists():
            agent["system_prompt"] = prompt_file.read_text(encoding="utf-8")
        elif not agent.get("system_prompt"):
            # Fallback a defaults
            default_prompt = DEFAULTS_DIR / "prompts" / f"{agent_id}.md"
            if default_prompt.exists():
                agent["system_prompt"] = default_prompt.read_text(encoding="utf-8")

        # 2. Leer contenido de cada skill file
        skill_contents = {}
        for skill_name in agent.get("skills", []):
            # Buscar el skill en DATA_DIR/prompts/skills/ o DEFAULTS_DIR/prompts/
            skill_paths = [
                DATA_DIR / "prompts" / "skills" / f"{skill_name}.md",
                DEFAULTS_DIR / "prompts" / f"{skill_name}.md",
                DATA_DIR / "prompts" / "system" / f"{skill_name}.md",
            ]
            for sp in skill_paths:
                if sp.exists():
                    skill_contents[skill_name] = sp.read_text(encoding="utf-8")
                    break
            else:
                skill_contents[skill_name] = ""  # skill no encontrado en disco

        agent["skill_contents"] = skill_contents

    return agents_data


@app.put("/api/agents/{agent_id}")
def update_agent(agent_id: str, update: AgentConfigUpdate):
    """Actualiza la configuración de un agente.
    Si se cambia el modelo y éste no está disponible en Ollama,
    inicia la descarga automática en background.
    """
    agents_file = DATA_DIR / "agents" / "agents.json"
    agents_data = load_json(agents_file, {"agents": []})

    for agent in agents_data.get("agents", []):
        if agent.get("id") == agent_id:
            if update.system_prompt is not None:
                # Guardar prompt en disco
                prompt_file = DATA_DIR / "prompts" / "system" / f"{agent_id}.md"
                prompt_file.write_text(update.system_prompt, encoding="utf-8")
                agent["system_prompt"] = update.system_prompt
            if update.model is not None:
                old_model = agent.get("model", "")
                new_model = update.model
                agent["model"] = new_model

                # Si el modelo cambio, verificar si está disponible y descargarlo si no
                if new_model != old_model:
                    if not _is_model_available(new_model):
                        logger.info(
                            f"Agente '{agent_id}': modelo '{new_model}' no disponible. "
                            f"Iniciando descarga automática..."
                        )
                        _start_pull_background(new_model)
                        agent["model_pull_status"] = "pulling"
                    else:
                        agent["model_pull_status"] = "available"

            if update.temperature is not None:
                agent["temperature"] = update.temperature
            agent["updated_at"] = datetime.utcnow().isoformat()
            save_json(agents_file, agents_data)

            # Incluir estado de descarga en la respuesta
            with _pull_lock:
                pull_info = _pull_status.get(agent.get("model", ""), {})
            agent["pull_status"] = pull_info if pull_info else None
            return agent

    raise HTTPException(status_code=404, detail="Agente no encontrado")


@app.put("/api/agents/{agent_id}/skills/{skill_name}")
def update_skill_content(agent_id: str, skill_name: str, body: dict):
    """Guarda el contenido editado de un skill file para un agente."""
    content = body.get("content", "")
    # Guardar en DATA_DIR/prompts/skills/{skill_name}.md
    skills_dir = DATA_DIR / "prompts" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skills_dir / f"{skill_name}.md"
    skill_file.write_text(content, encoding="utf-8")
    return {"ok": True, "skill": skill_name, "agent_id": agent_id, "size": len(content)}


@app.get("/api/audit")
def get_audit_log(date: Optional[str] = None, agent_id: Optional[str] = None):
    """Retorna el log de auditoría filtrado."""
    audit_dir = DATA_DIR / "audit"
    entries = []

    if date:
        audit_file = audit_dir / f"{date}.jsonl"
        if audit_file.exists():
            for line in audit_file.read_text().splitlines():
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    else:
        # Últimas 24h
        for audit_file in sorted(audit_dir.glob("*.jsonl"), reverse=True)[:3]:
            for line in audit_file.read_text().splitlines():
                if line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if agent_id:
        entries = [e for e in entries if e.get("agent_id") == agent_id]

    return {"entries": entries[:100], "total": len(entries)}


# ---------------------------------------------------------------------------
# Servir la UI
# ---------------------------------------------------------------------------
if APP_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(APP_DIR), html=True), name="ui")
else:
    logger.warning(f"Directorio UI no encontrado en {APP_DIR}")


@app.get("/")
def root():
    """Redirige a la UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/index.html")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # En Windows, escuchar en 127.0.0.1 para evitar que el browser abra 0.0.0.0
    # En Linux/macOS, escuchar en 0.0.0.0 para acceso desde red local
    host = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"
    uvicorn.run(app, host=host, port=PORT, log_level="info")

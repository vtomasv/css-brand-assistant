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
import json
import uuid
import shutil
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
        config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    logger.info(f"CSS Brand Assistant iniciado. DATA_DIR={DATA_DIR}")

# ---------------------------------------------------------------------------
# Utilidades de persistencia
# ---------------------------------------------------------------------------
def save_json(path: Path, data: Any) -> None:
    """Guarda datos como JSON con formato legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


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
def call_ollama(model: str, system_prompt: str, user_message: str,
                temperature: float = 0.7, timeout: int = 120) -> str:
    """Llama al LLM local vía Ollama API."""
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
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama no está disponible. Asegúrate de que Ollama esté corriendo.",
        )
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

        result = call_ollama(model, system_prompt, user_message, temperature=0.3)
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
        soup = BeautifulSoup(resp.text, "html.parser")

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

    response = call_ollama(model, system_prompt, user_message, temperature=0.7)
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
        result = call_ollama(model, system_prompt, user_message, temperature=0.3, timeout=180)
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
    """Genera la planificación temporal y publicaciones de la campaña."""
    import time
    start = time.time()
    campaign_dir = DATA_DIR / "campaigns" / f"{brand_id}_{campaign_id}"

    try:
        model = get_active_model()
        adn_summary = json.dumps(adn.get("fields", {}), ensure_ascii=False)[:3000]

        # Paso 1: Generar estructura narrativa de la campaña
        system_prompt = get_system_prompt("campaign_strategist") or _get_campaign_strategist_prompt()
        user_message = f"""Crea la planificación estratégica para esta campaña:

CAMPAÑA: {campaign_data['name']}
OBJETIVO: {campaign_data['objective']}
PRODUCTO/TEMA: {campaign_data['product_or_topic']}
AUDIENCIA: {campaign_data['target_audience']}
PERÍODO: {campaign_data['start_date']} al {campaign_data['end_date']}
CANALES: {', '.join(campaign_data['channels'])}
FRECUENCIA: {campaign_data['frequency']}

ADN DE MARCA:
{adn_summary}

Genera un plan con etapas narrativas, distribución por canal y calendario de publicaciones.
Responde en JSON con la estructura: stages (lista de etapas) y publications (lista de publicaciones)."""

        plan_result = call_ollama(model, system_prompt, user_message, temperature=0.5)

        # Parsear y guardar plan
        plan = _parse_campaign_plan(plan_result, campaign_data)
        save_json(campaign_dir / "plan.json", plan)

        # Actualizar estado de la campaña
        camp_file = campaign_dir / "campaign.json"
        camp = load_json(camp_file)
        camp["status"] = "active"
        camp["publications_count"] = len(plan.get("publications", []))
        camp["stages_count"] = len(plan.get("stages", []))
        camp["updated_at"] = datetime.utcnow().isoformat()
        save_json(camp_file, camp)

        latency = int((time.time() - start) * 1000)
        log_audit("campaign_strategist", "generate_campaign_plan",
                  {"brand_id": brand_id, "campaign_id": campaign_id},
                  plan_result[:500], model, latency, True)
        logger.info(f"Plan de campaña generado: {campaign_id} — {len(plan.get('publications', []))} publicaciones")

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


def _parse_campaign_plan(llm_output: str, campaign_data: dict) -> dict:
    """Parsea el plan de campaña del LLM, con fallback a estructura básica."""
    import re
    from datetime import date

    json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "publications" in parsed:
                # Asegurar que cada publicación tenga ID y campos requeridos
                for i, pub in enumerate(parsed["publications"]):
                    if "id" not in pub:
                        pub["id"] = str(uuid.uuid4())
                    pub.setdefault("status", "pending")
                    pub.setdefault("campaign_id", campaign_data["id"])
                    pub.setdefault("brand_id", campaign_data["brand_id"])
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: generar publicaciones básicas
    start = datetime.strptime(campaign_data["start_date"], "%Y-%m-%d")
    end = datetime.strptime(campaign_data["end_date"], "%Y-%m-%d")
    days = (end - start).days + 1
    channels = campaign_data.get("channels", ["Instagram"])

    publications = []
    stages = [
        {"name": "Descubrimiento", "description": "Presentación y awareness", "days": "1-3"},
        {"name": "Consideración", "description": "Beneficios y propuesta de valor", "days": "4-8"},
        {"name": "Activación", "description": "CTA directo y conversión", "days": "9-12"},
        {"name": "Cierre", "description": "Urgencia y recordación", "days": "13+"},
    ]

    pub_count = 0
    for day_offset in range(min(days, 15)):
        current_date = start + timedelta(days=day_offset)
        stage_idx = min(day_offset // 4, len(stages) - 1)

        for channel in channels[:2]:  # máximo 2 canales en fallback
            pub_count += 1
            publications.append({
                "id": str(uuid.uuid4()),
                "campaign_id": campaign_data["id"],
                "brand_id": campaign_data["brand_id"],
                "channel": channel,
                "scheduled_at": current_date.strftime("%Y-%m-%d") + " 10:00",
                "stage": stages[stage_idx]["name"],
                "objective": campaign_data["objective"],
                "text": f"[Publicación {pub_count} — {channel} — {stages[stage_idx]['name']}]\n{llm_output[:200]}",
                "hashtags": ["#marca", "#marketing", "#pyme"],
                "cta": "¡Contáctanos!",
                "image_prompt": f"Imagen para {channel} sobre {campaign_data['product_or_topic']}",
                "status": "pending",
                "edit_status": "draft",
            })

    return {"stages": stages, "publications": publications, "raw_plan": llm_output[:2000]}


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
                                  language: Optional[str] = None):
    """Regenera una publicación con instrucción y/o idioma opcionales."""
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

                    model = get_active_model()
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

                    result = call_ollama(model, system_prompt, user_message, temperature=0.8)
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


@app.post("/api/campaigns/{campaign_id}/publications/{pub_id}/generate-image")
async def generate_publication_image(campaign_id: str, pub_id: str, req: GenerateImageRequest):
    """Genera una imagen para la publicación usando Ollama con modelo de imagen.
    Si el modelo no está disponible, lo descarga automáticamente antes de generar.
    """
    import time, base64
    start = time.time()

    try:
        # 1. Verificar y descargar el modelo si es necesario
        model_status = _ensure_model_available(req.model)
        if not model_status["available"]:
            return {
                "error": f"No se pudo preparar el modelo {req.model}: {model_status['error']}",
                "success": False,
            }
        if model_status["pulled"]:
            logger.info(f"Modelo {req.model} descargado exitosamente antes de generar")

        # 2. Construir prompt
        prompt = req.image_prompt
        if req.instruction:
            prompt = f"{prompt}. Estilo adicional: {req.instruction}"

        # 3. Llamar a Ollama via POST /api/generate (endpoint oficial para modelos de imagen)
        # NOTA: La documentacion oficial de Ollama indica que para modelos de imagen
        # se usa /api/generate con stream=False. La respuesta final tiene el campo 'image'
        # con el base64 de la imagen generada.
        # Ref: https://github.com/ollama/ollama/blob/main/docs/api.md#image-generation-experimental
        logger.info(f"Generando imagen con modelo {req.model}, prompt: {prompt[:100]}...")
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": req.model,
                "prompt": prompt,
                "stream": False,
                "width": 1024,
                "height": 1024,
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        # Log completo de la respuesta para diagnostico
        response_keys = list(data.keys())
        logger.info(f"Ollama /api/generate response keys: {response_keys}")
        for key in response_keys:
            val = data[key]
            if isinstance(val, str) and len(val) > 100:
                logger.info(f"  {key}: [string len={len(val)}] {val[:80]}...")
            else:
                logger.info(f"  {key}: {repr(val)[:200]}")

        # 4. Extraer imagen base64
        # La respuesta final tiene el campo 'image' con el base64 (formato Ollama nativo)
        # Tambien intentamos 'images' (array) y 'data[0].b64_json' (formato OpenAI) como fallback
        image_b64 = None

        if "image" in data and data["image"]:
            # Formato oficial Ollama: campo 'image' con base64 directo
            image_b64 = data["image"]
            logger.info(f"Imagen encontrada en campo 'image', longitud: {len(image_b64)}")
        elif "images" in data and data["images"]:
            # Formato alternativo: array 'images'
            image_b64 = data["images"][0]
            logger.info(f"Imagen encontrada en campo 'images[0]', longitud: {len(image_b64)}")
        elif "data" in data and data["data"]:
            # Formato OpenAI-compatible
            first_item = data["data"][0]
            if "b64_json" in first_item and first_item["b64_json"]:
                image_b64 = first_item["b64_json"]
                logger.info(f"Imagen encontrada en data[0].b64_json, longitud: {len(image_b64)}")
            elif "url" in first_item and first_item["url"]:
                img_url = first_item["url"]
                logger.info(f"Imagen como URL: {img_url}, descargando...")
                img_resp = requests.get(img_url, timeout=60)
                img_resp.raise_for_status()
                import base64 as _b64
                image_b64 = _b64.b64encode(img_resp.content).decode("utf-8")
                logger.info(f"Imagen descargada y convertida a base64, longitud: {len(image_b64)}")
        elif "response" in data and data["response"]:
            # Ultimo fallback: 'response' podria ser base64 en versiones antiguas
            candidate = data["response"].strip()
            import re as _re
            if bool(_re.match(r'^[A-Za-z0-9+/=\s]{500,}$', candidate)):
                image_b64 = candidate.replace('\n', '').replace('\r', '').replace(' ', '')
                logger.info(f"Imagen en campo 'response' (base64), longitud: {len(image_b64)}")
            else:
                logger.warning(f"Campo 'response' no es base64. Preview: {repr(candidate[:200])}")
        else:
            logger.error(f"Ningun campo de imagen encontrado. Keys disponibles: {response_keys}")
            logger.error(f"Respuesta completa: {str(data)[:1000]}")

        # Tambien intentar con /v1/images/generations (OpenAI-compatible) si no hubo imagen
        if not image_b64:
            logger.warning("Intentando fallback con /v1/images/generations (OpenAI-compatible)...")
            try:
                fallback_resp = requests.post(
                    f"{OLLAMA_URL}/v1/images/generations",
                    json={
                        "model": req.model,
                        "prompt": prompt,
                        "size": "1024x1024",
                        "response_format": "b64_json",
                        "n": 1,
                    },
                    timeout=300,
                )
                if fallback_resp.status_code == 200:
                    fb_data = fallback_resp.json()
                    logger.info(f"Fallback /v1/images/generations keys: {list(fb_data.keys())}")
                    if "data" in fb_data and fb_data["data"]:
                        first = fb_data["data"][0]
                        if "b64_json" in first and first["b64_json"]:
                            image_b64 = first["b64_json"]
                            logger.info(f"Imagen en fallback OpenAI, longitud: {len(image_b64)}")
                else:
                    logger.warning(f"Fallback /v1/images/generations status: {fallback_resp.status_code}")
            except Exception as fb_err:
                logger.warning(f"Fallback /v1/images/generations error: {fb_err}")

        if image_b64:
            # 5. Guardar imagen en disco con timestamp para evitar caché
            img_dir = DATA_DIR / "exports" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            img_filename = f"{pub_id}_{ts}.png"
            img_path = img_dir / img_filename

            try:
                img_bytes = base64.b64decode(image_b64, validate=False)
                img_path.write_bytes(img_bytes)
                logger.info(f"Imagen guardada: {img_path} ({len(img_bytes)} bytes)")
            except Exception as decode_err:
                logger.error(f"Error decodificando base64: {decode_err}")
                return {"error": f"Error decodificando imagen: {str(decode_err)}", "success": False}

            # También guardar como nombre fijo (para referencia persistente)
            fixed_path = img_dir / f"{pub_id}.png"
            fixed_path.write_bytes(img_bytes)

            # 6. Actualizar publicación con URL de imagen
            image_url = f"/api/images/{pub_id}.png?t={ts}"
            pub_updated = False

            # Buscar el plan.json de la campaña por directorio exacto o por búsqueda
            # El directorio se llama {brand_id}_{campaign_id}
            campaigns_root = DATA_DIR / "campaigns"
            if campaigns_root.exists():
                # Primero intentar ruta directa (más eficiente)
                for camp_dir in campaigns_root.iterdir():
                    if not camp_dir.is_dir():
                        continue
                    # El nombre puede ser brand_id_campaign_id o solo campaign_id
                    dir_name = camp_dir.name
                    if campaign_id in dir_name or dir_name == campaign_id:
                        plan_file = camp_dir / "plan.json"
                        if not plan_file.exists():
                            logger.warning(f"plan.json no encontrado en {camp_dir}")
                            continue
                        plan = load_json(plan_file, {"publications": []})
                        for pub in plan.get("publications", []):
                            if pub.get("id") == pub_id:
                                pub["generated_image_url"] = f"/api/images/{pub_id}.png"
                                pub["updated_at"] = datetime.utcnow().isoformat()
                                save_json(plan_file, plan)
                                pub_updated = True
                                logger.info(f"Publicación {pub_id} actualizada con imagen en {plan_file}")
                                break
                    if pub_updated:
                        break

            if not pub_updated:
                logger.warning(f"No se encontró la publicación {pub_id} en campaña {campaign_id} para actualizar imagen")

            latency = int((time.time() - start) * 1000)
            log_audit("image_generator", "generate_image",
                      {"campaign_id": campaign_id, "pub_id": pub_id, "model": req.model},
                      f"Image generated: {img_filename} ({len(img_bytes)} bytes)", req.model, latency, True)

            # Devolver base64 directamente para que el frontend pueda mostrar la imagen
            # sin depender del endpoint de archivos (evita problemas de URL relativa/absoluta)
            return {
                "image_url": image_url,
                "image_filename": img_filename,
                "image_size_bytes": len(img_bytes),
                "image_b64": image_b64,  # base64 directo para mostrar como data:image/png;base64,...
                "success": True,
            }
        else:
            # El modelo respondió pero sin imagen — devolver debug info completo
            debug_info = {
                "response_keys": list(data.keys()),
                "data_count": len(data.get("data", [])),
                "data_preview": str(data.get("data", []))[:300] if data.get("data") else None,
                "error_detail": data.get("error", {}) if isinstance(data.get("error"), dict) else data.get("error"),
            }
            logger.error(f"Modelo {req.model} no retornó imagen. Debug: {debug_info}")
            return {
                "error": (
                    f"El modelo {req.model} no retornó una imagen válida. "
                    "Verifica que Ollama esté en versión >= 0.5.4 y que el modelo sea compatible con generación de imágenes. "
                    "Nota: la generación de imágenes solo está disponible en macOS actualmente."
                ),
                "debug": debug_info,
                "success": False,
            }

    except requests.exceptions.ConnectionError:
        return {"error": "Ollama no está disponible", "success": False}
    except Exception as e:
        logger.error(f"Error generando imagen: {e}")
        return {"error": str(e), "success": False}


@app.get("/api/images/{filename}")
def serve_generated_image(filename: str):
    """Sirve imágenes generadas por IA con headers anti-caché."""
    # Soportar filename con query string (ej: pub_id.png?t=123)
    clean_filename = filename.split("?")[0]
    img_path = DATA_DIR / "exports" / "images" / clean_filename
    if not img_path.exists():
        logger.warning(f"Imagen no encontrada: {img_path}")
        raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {clean_filename}")
    return FileResponse(
        str(img_path),
        media_type="image/png",
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
    """Actualiza la configuración de un agente."""
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
                agent["model"] = update.model
            if update.temperature is not None:
                agent["temperature"] = update.temperature
            agent["updated_at"] = datetime.utcnow().isoformat()
            save_json(agents_file, agents_data)
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
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

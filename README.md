# CSS Brand Assistant

**Plugin de Pinokio para gestión de ADN de marca y campañas digitales con IA local para PYMEs**

![CSS Brand Assistant](icon.png)

CSS Brand Assistant es un plugin para [Pinokio](https://pinokio.computer) que permite a las pequeñas y medianas empresas construir una identidad de marca estructurada —denominada **ADN empresarial**— y utilizarla como base para planificar y generar campañas de marketing digital multicanal, todo con inteligencia artificial ejecutándose localmente en la computadora del usuario, sin dependencia de servicios en la nube.

---

## Características principales

El plugin implementa el ciclo completo de marketing asistido por IA descrito en la especificación funcional del proyecto CCS:

**Módulo de Marcas.** Permite registrar y gestionar múltiples marcas con sus datos básicos: nombre, sitio web, sector, mercados objetivo e idioma. Cada marca tiene su propio espacio de trabajo semiaislado con historial, ADN y campañas independientes.

**Análisis automático de sitio web.** A partir de la URL de la marca, el agente analizador extrae señales de identidad: propuesta de valor, tono comunicacional, paleta de colores, estilo visual, productos y servicios, y público objetivo. El proceso se ejecuta en background y produce un borrador de ADN automáticamente.

**Entrevista guiada por agente.** Un agente especializado en marketing conduce una conversación de descubrimiento para refinar el ADN. Las preguntas son contextuales, progresivas y ancladas en lo que el sistema ya sabe de la marca. La sesión es pausable y retomable.

**ADN empresarial versionado.** El ADN es un objeto semiestructurado con campos claros: propuesta de valor, tono, personalidad, paleta, tipografía, público objetivo, diferenciadores, restricciones y más. Cada aprobación genera una nueva versión, preservando la trazabilidad para campañas existentes.

**Generación de campañas con IA.** El agente estratega toma el ADN y los objetivos de negocio para construir una planificación temporal con etapas narrativas (descubrimiento, consideración, activación, cierre) y publicaciones concretas por canal.

**Calendario de publicaciones.** Vista operativa de todas las publicaciones de una campaña, filtrable por canal, con estado de publicación manual (pendiente, lista, publicada, omitida, reprogramada).

**Editor de publicaciones.** Pantalla dedicada para editar texto, hashtags, CTA y prompt de imagen de cada publicación. Incluye simulación visual del post y regeneración asistida con instrucción libre.

**Configuración de agentes.** Panel para editar los system prompts de cada agente directamente desde la UI, sin necesidad de reiniciar el servidor.

**Auditoría completa.** Registro de todas las operaciones de agentes: qué agente ejecutó cada tarea, qué modelo usó, qué entradas recibió, qué salida produjo y cuánto tardó.

---

## Arquitectura

```
css-brand-assistant/
├── pinokio.js          # Configuración y menú dinámico del plugin
├── install.json        # Instalación con 1 click (Ollama + venv + deps)
├── start.json          # Inicio del servidor como daemon
├── stop.json           # Parada del servidor
├── reset.json          # Desinstalación (conserva datos)
├── requirements.txt    # Dependencias Python
├── icon.png            # Ícono 512x512
├── app/
│   └── index.html      # Frontend SPA (HTML + CSS + JS vanilla)
├── server/
│   └── app.py          # Backend FastAPI con todos los módulos
├── defaults/
│   ├── agents.json     # Configuración inicial de agentes
│   └── prompts/        # System prompts por defecto
│       ├── brand_analyzer.md
│       ├── brand_interviewer.md
│       └── campaign_strategist.md
└── data/               # Datos del usuario (persistentes)
    ├── config.json
    ├── agents/
    ├── prompts/system/
    ├── brands/
    ├── campaigns/
    ├── sessions/
    ├── exports/
    └── audit/
```

### Agentes de IA

| Agente | Rol | Modelo recomendado | Función |
|--------|-----|-------------------|---------|
| `brand_analyzer` | Analizador | `llama3.2:3b` | Analiza sitios web e infiere identidad de marca |
| `brand_interviewer` | Entrevistador | `llama3.1:8b` | Conduce entrevistas de descubrimiento de marca |
| `campaign_strategist` | Estratega | `llama3.1:8b` | Diseña planificaciones temporales de campañas |
| `content_writer` | Redactor | `llama3.2:3b` | Genera y mejora publicaciones para redes sociales |

### API REST

El backend expone los siguientes dominios de API:

| Dominio | Endpoints principales |
|---------|----------------------|
| Sistema | `GET /api/health`, `GET /api/config`, `GET /api/ollama/status` |
| Marcas | `GET/POST /api/brands`, `GET/PUT/DELETE /api/brands/{id}` |
| Análisis web | `POST /api/brands/{id}/analyze-website` |
| ADN | `GET /api/brands/{id}/adn`, `PUT /api/brands/{id}/adn/field`, `POST /api/brands/{id}/adn/approve` |
| Entrevista | `POST /api/brands/{id}/interview` |
| Campañas | `GET/POST /api/brands/{id}/campaigns` |
| Publicaciones | `GET/PUT /api/campaigns/{id}/publications`, `POST .../regenerate` |
| Agentes | `GET /api/agents`, `PUT /api/agents/{id}` |
| Auditoría | `GET /api/audit` |

---

## Requisitos del sistema

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| RAM | 4 GB | 8 GB o más |
| Almacenamiento | 5 GB libres | 10 GB libres |
| Python | 3.9+ | 3.11+ |
| Pinokio | Última versión | Última versión |
| Ollama | Instalado | Instalado |

El modelo de IA se selecciona automáticamente según la RAM disponible:

| RAM disponible | Modelo instalado |
|---------------|-----------------|
| Menos de 6 GB | `llama3.2:1b` |
| 6 a 12 GB | `llama3.2:3b` |
| Más de 12 GB | `llama3.1:8b` |

---

## Instalación

### Opción 1: Instalación desde Pinokio (recomendada)

1. Abre Pinokio en tu computadora.
2. Ve a **Discover** o **Install from URL**.
3. Ingresa la URL del repositorio: `https://github.com/vtomasv/css-brand-assistant`
4. Haz clic en **Instalar** y espera a que el proceso termine automáticamente.

### Opción 2: Instalación manual

```bash
# Clonar el repositorio en el directorio de Pinokio
cd ~/pinokio/api
git clone https://github.com/vtomasv/css-brand-assistant

# El plugin aparecerá automáticamente en Pinokio
```

---

## Uso básico

El flujo de trabajo recomendado para una nueva marca es el siguiente:

**Paso 1 — Crear la marca.** Ve a la sección Marcas y haz clic en "Nueva Marca". Ingresa el nombre, sitio web y sector de la empresa.

**Paso 2 — Analizar el sitio web.** Desde la tarjeta de la marca, haz clic en "Analizar web". El agente analizador procesará el sitio y construirá un borrador de ADN automáticamente. Este proceso puede tomar entre 1 y 5 minutos dependiendo del hardware.

**Paso 3 — Entrevista de descubrimiento.** Ve a la sección "Entrevista IA" y responde las preguntas del agente de marketing. Cada respuesta refina el ADN de la marca.

**Paso 4 — Aprobar el ADN.** Cuando el ADN esté completo, ve a "ADN de Marca" y haz clic en "Aprobar ADN". Esto crea una versión oficial que será usada por todas las campañas.

**Paso 5 — Crear una campaña.** Ve a "Campañas" y haz clic en "Nueva Campaña". Completa el formulario con objetivo, fechas, canales y audiencia. La IA generará la planificación temporal y las publicaciones automáticamente.

**Paso 6 — Revisar y publicar.** Abre el calendario para ver todas las publicaciones. Haz clic en cualquier publicación para editarla, copiar el texto o marcarla como publicada.

---

## Plan de desarrollo detallado

### Versión 0.1 (actual) — Funcionalidades base

La versión actual implementa el núcleo funcional del sistema:

- Gestión completa de marcas (CRUD)
- Análisis automático de sitio web con agente LLM
- Entrevista conversacional de descubrimiento de marca
- Construcción y versionado del ADN empresarial
- Creación de campañas con planificación temporal generada por IA
- Calendario de publicaciones con filtros por canal
- Editor de publicaciones con regeneración asistida
- Configuración de agentes y edición de prompts
- Auditoría completa de operaciones de agentes

### Versión 0.2 — Mejoras de contenido

Las siguientes funcionalidades están planificadas para la próxima iteración:

- Generación local de imágenes con Stable Diffusion (via Ollama o AUTOMATIC1111)
- Exportación de publicaciones en PDF y CSV
- Vista de calendario mensual con drag & drop
- Variantes de copy para cada publicación
- Análisis de consistencia del ADN entre publicaciones

### Versión 0.3 — Colaboración y análisis

- Soporte para múltiples usuarios con roles (administrador, responsable de marca, operador)
- Dashboard de seguimiento con métricas de publicaciones realizadas vs. planificadas
- Biblioteca de activos visuales por marca
- Clonación de campañas exitosas
- A/B testing de copys

### Versión 1.0 — Integración y automatización

- Integración opcional con APIs de redes sociales para publicación directa
- Analítica de performance real (alcance, engagement, conversiones)
- Aprendizaje basado en resultados históricos
- Soporte multiidioma completo (español, inglés, portugués)
- Instalador para Windows, macOS y Linux con detección automática de hardware

---

## Contribuir

Este plugin es parte del proyecto CCS (Plugins para Pinokio). Las contribuciones son bienvenidas.

Para reportar errores o sugerir mejoras, abre un issue en el repositorio de GitHub.

---

## Licencia

MIT License — Ver archivo `LICENSE` para más detalles.

---

*Desarrollado como parte del proyecto CCS — Plugins de IA local para PYMEs.*

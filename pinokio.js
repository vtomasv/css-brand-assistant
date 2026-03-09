/**
 * CSS Brand Assistant — Configuración de Plugin Pinokio
 *
 * Menú dinámico según estado del plugin:
 *   - No instalado: botón de instalación
 *   - Instalado y corriendo: estado activo + botón detener + abrir UI
 *   - Instalado y detenido: botón iniciar
 */
module.exports = {
  title: "CSS Brand Assistant",
  description: "Plataforma de ADN de marca y campañas digitales con IA local para PYMEs",
  icon: "icon.png",

  menu: async (kernel, info) => {
    // Verificar si el plugin está instalado (venv creado)
    const installed = await kernel.exists(__dirname, "venv")

    if (!installed) {
      return [
        {
          default: true,
          icon: "fa-solid fa-download",
          text: "Instalar",
          href: "install.json",
        },
      ]
    }

    // Verificar si el servidor está corriendo
    const running = await kernel.script.running(__dirname, "start.json")

    if (running) {
      return [
        {
          icon: "fa-solid fa-circle",
          text: "En ejecución",
          href: "start.json",
          style: "color: #22c55e",
        },
        {
          icon: "fa-solid fa-arrow-up-right-from-square",
          text: "Abrir UI",
          href: "http://localhost:{{port}}/ui/index.html",
        },
        {
          icon: "fa-solid fa-stop",
          text: "Detener",
          href: "stop.json",
        },
      ]
    }

    return [
      {
        default: true,
        icon: "fa-solid fa-play",
        text: "Iniciar",
        href: "start.json",
      },
      {
        icon: "fa-solid fa-trash",
        text: "Desinstalar",
        href: "reset.json",
      },
    ]
  },
}

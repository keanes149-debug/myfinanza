"""Guion, colores y rutas compartidas por voz.py, audio.py y escenas.py."""
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
BUILD = os.environ.get("NOVARA_BUILD", os.path.join(AQUI, "build"))
MODELOS = os.environ.get("NOVARA_MODELOS", os.path.join(AQUI, "modelos"))
FUENTE = os.environ.get("NOVARA_FUENTE", os.path.join(AQUI, "fuentes", "DMSans.ttf"))

W, H, FPS = 1080, 1920, 30
TRANSICION = 0.3          # segundos de fundido / deslizamiento al inicio de cada escena
VOZ_RETARDO = 0.30        # la voz arranca este tiempo despues del inicio de la escena
VOZ = "em_alex"
VELOCIDAD = 1.15
IDIOMA = "es-419"         # espanol latinoamericano: seseo ("sita", no "zita")

# Pausas entre frases (segundos)
PAUSA_COMA = 0.12
PAUSA_PUNTO = 0.30
PAUSA_SUSPENSIVOS = 0.38

VERDE = (0x1F, 0x6B, 0x48)
VERDE_CLARO = (0xE1, 0xEF, 0xE6)
CREMA = (0xF3, 0xF0, 0xE8)
TEXTO = (0x1C, 0x23, 0x1F)
BLANCO = (0xFF, 0xFF, 0xFF)
GRIS = (0x6E, 0x7A, 0x73)
GRIS_CLARO = (0xD9, 0xD6, 0xCC)
ROJO = (0xC9, 0x4A, 0x3F)
VERDE_OSCURO = (0x16, 0x50, 0x36)
VERDE_MEDIO = (0x3E, 0x95, 0x6A)

# Cada escena: duracion minima (la del guion) y la frase de voz.
# "tts" es el texto que se le pasa a Kokoro (escrito como se pronuncia);
# "voz" es el texto real, solo para referencia.
ESCENAS = [
    dict(id=1, nombre="gancho", minimo=5.0, transicion="fundido",
         voz="¿Cuántos clientes pierdes por no contestar el WhatsApp a tiempo?",
         tts="¿Cuántos clientes pierdes por no contestar el guátsap a tiempo?"),
    dict(id=2, nombre="problema", minimo=6.0, transicion="deslizar",
         voz="Mientras atiendes, o cuando ya cerraste, tus clientes escriben… y si nadie responde, se van a otro lado.",
         tts="Mientras atiendes, o cuando ya cerraste, tus clientes escriben... y si nadie responde, se van a otro lado."),
    dict(id=3, nombre="solucion", minimo=4.0, transicion="fundido",
         voz="Con Novara, tu WhatsApp responde y agenda solo, las 24 horas.",
         tts="Con Novara, tu guátsap responde y agenda solo, las veinticuatro horas."),
    dict(id=4, nombre="paso1", minimo=7.0, transicion="deslizar",
         voz="El cliente pregunta, el asistente revisa tu agenda y le ofrece solo horarios libres.",
         tts="El cliente pregunta, el asistente revisa tu agenda y le ofrece solo horarios libres."),
    dict(id=5, nombre="paso2", minimo=7.0, transicion="fundido",
         voz="Si quiere cambiar la hora, la cita vieja se libera y la nueva queda agendada. Sin cruces, sin doble reserva.",
         tts="Si quiere cambiar la hora, la cita vieja se libera y la nueva queda agendada. Sin cruces, sin doble reserva."),
    dict(id=6, nombre="panel", minimo=8.0, transicion="deslizar",
         voz="Tú ves toda tu agenda del día en un panel. Las citas que anotas a mano también se respetan, y puedes leer cada conversación.",
         tts="Tú ves toda tu agenda del día en un panel. Las citas que anotas a mano también se respetan, y puedes leer cada conversación."),
    dict(id=7, nombre="feriados", minimo=6.0, transicion="deslizar",
         voz="¿Feriado o viaje? Se lo dices al asistente y deja de agendar ese día.",
         tts="¿Feriado o viaje? Se lo dices al asistente y deja de agendar ese día."),
    dict(id=8, nombre="cierre", minimo=7.0, transicion="fundido",
         voz="Novara Software. Pide tu demostración gratis.",
         tts="Novara Software. Pide tu demostración gratis."),
]

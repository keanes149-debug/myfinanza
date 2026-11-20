# Video explicativo Novara — "Cómo funciona"

`novara-como-funciona.mp4`: 1080×1920 vertical, 30 fps, H.264 + AAC, 53 s, español (Bolivia).
Todo se genera con código: cuadros dibujados con Pillow, voz con Kokoro (TTS local), música con numpy, unido con ffmpeg.

## Archivos

| Archivo | Qué hace |
|---|---|
| `guion.py` | Guion por escena (texto en pantalla y texto para la voz), colores de marca, rutas |
| `voz.py` | Voz en off con Kokoro `em_alex`, velocidad 1.15, `es-419` (seseo). Sintetiza frase por frase para conocer el tiempo exacto de cada una |
| `escenas.py` | Calcula la duración de cada escena según su voz, dibuja las 8 escenas y las codifica (una escena = un MP4) |
| `audio.py` | Música 118 BPM generada con numpy, efectos de sonido, ducking bajo la voz y loudnorm a −14 LUFS (2 pasadas) |
| `hacer_video.py` | Ejecuta todo: voz → plan → cuadros de revisión → render → concat → audio → MP4 final |

## Regenerar

```bash
pip install kokoro-onnx soundfile pillow numpy        # además: ffmpeg
mkdir -p modelos fuentes
curl -L -o modelos/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o modelos/voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
curl -L -o fuentes/DMSans.ttf "https://raw.githubusercontent.com/google/fonts/main/ofl/dmsans/DMSans%5Bopsz%2Cwght%5D.ttf"
python3 hacer_video.py
```

Los cuadros de revisión (uno por escena, en el momento con más elementos) quedan en `build/fotos/`.

Notas de pronunciación: en el texto para la voz, "WhatsApp" se escribe `guátsap` y "24" se escribe `veinticuatro`.

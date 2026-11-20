#!/usr/bin/env python3
"""Genera novara-como-funciona.mp4 de punta a punta.

  1. voz.py      voz en off por escena (Kokoro, em_alex, es-419) + tiempos de cada frase
  2. escenas.py  plan de tiempos (cada escena dura lo que su frase) y render de cada escena a MP4
  3. ffmpeg      concat de las 8 escenas (mismas medidas y ajustes, sin recodificar)
  4. audio.py    musica numpy + efectos + voz + ducking + loudnorm (2 pasadas)
  5. ffmpeg      une video y audio (H.264 + AAC) y verifica el resultado

Rutas por variables de entorno (ver guion.py): NOVARA_BUILD, NOVARA_MODELOS, NOVARA_FUENTE.
"""
import json
import os
import subprocess
import sys
import time

from guion import AQUI, BUILD, ESCENAS

SALIDA = os.environ.get("NOVARA_SALIDA", os.path.join(AQUI, "novara-como-funciona.mp4"))


def paso(nombre, cmd):
    t = time.time()
    print(f"\n== {nombre}", flush=True)
    subprocess.run(cmd, check=True, cwd=AQUI)
    print(f"   ({time.time() - t:.1f} s)", flush=True)


def main():
    py = sys.executable
    if "--sin-voz" not in sys.argv or not os.path.exists(os.path.join(BUILD, "voz", "tiempos.json")):
        paso("voz en off (Kokoro)", [py, "voz.py"])
    paso("plan de tiempos", [py, "escenas.py", "plan"])
    paso("cuadros de revision", [py, "escenas.py", "fotos"])
    paso("render de escenas", [py, "escenas.py", "render"])

    lista = os.path.join(BUILD, "escenas", "lista.txt")
    with open(lista, "w") as f:
        for e in ESCENAS:
            f.write(f"file 'escena_{e['id']}.mp4'\n")
    video = os.path.join(BUILD, "video_sin_audio.mp4")
    paso("concat de escenas", ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lista,
                               "-c", "copy", video])
    paso("audio: musica, efectos, voz, ducking, loudnorm", [py, "audio.py"])
    paso("video final", ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                         "-i", os.path.join(BUILD, "audio", "mezcla_final.wav"),
                         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                         "-ar", "48000", "-shortest", "-movflags", "+faststart", SALIDA])

    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", SALIDA],
                                     capture_output=True, text=True, check=True).stdout)
    for s in info["streams"]:
        if s["codec_type"] == "video":
            print(f"video: {s['codec_name']} {s['width']}x{s['height']} {s['r_frame_rate']} fps, "
                  f"{s['pix_fmt']}, {float(s['duration']):.2f} s, {s.get('nb_frames')} cuadros")
        else:
            print(f"audio: {s['codec_name']} {s['sample_rate']} Hz {s['channels']} canales, {float(s['duration']):.2f} s")
    print(f"archivo: {SALIDA} ({int(info['format']['size']) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Genera la voz en off por escena con Kokoro (em_alex) y guarda los tiempos de cada frase.

Kokoro (este modelo) no devuelve tiempos por fonema, asi que cada linea se sintetiza
frase por frase (cortando en comas, puntos y puntos suspensivos) y se une con pausas
controladas. Asi se sabe exactamente cuando empieza cada frase para sincronizar la imagen.

Salida: build/voz/escena_N.wav (24 kHz mono) y build/voz/tiempos.json
"""
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

from guion import (BUILD, ESCENAS, IDIOMA, MODELOS, PAUSA_COMA, PAUSA_PUNTO,
                   PAUSA_SUSPENSIVOS, VELOCIDAD, VOZ)


def frases(texto):
    """Corta el texto en frases, cada una con su signo final y la pausa que le sigue."""
    partes = re.findall(r"[^,.?!]+(?:\.\.\.|[,.?!])?", texto)
    out = []
    for p in partes:
        p = p.strip()
        if not p:
            continue
        if p.endswith("..."):
            pausa = PAUSA_SUSPENSIVOS
        elif p.endswith(","):
            pausa = PAUSA_COMA
        else:
            pausa = PAUSA_PUNTO
        out.append((p, pausa))
    return out


def recortar(a, sr, umbral_db=-42):
    """Quita silencio al inicio y al final (dejando 15 ms de colchon)."""
    marco = int(sr * 0.01)
    n = len(a) // marco
    rms = np.sqrt((a[: n * marco].reshape(n, marco) ** 2).mean(1) + 1e-12)
    activo = np.where(rms > rms.max() * 10 ** (umbral_db / 20))[0]
    if len(activo) == 0:
        return a
    i0 = max(0, activo[0] * marco - int(sr * 0.015))
    i1 = min(len(a), (activo[-1] + 1) * marco + int(sr * 0.015))
    return a[i0:i1]


def main():
    from kokoro_onnx import Kokoro

    k = Kokoro(os.path.join(MODELOS, "kokoro-v1.0.onnx"), os.path.join(MODELOS, "voices-v1.0.bin"))
    salida = os.path.join(BUILD, "voz")
    os.makedirs(salida, exist_ok=True)
    tiempos = {}
    for esc in ESCENAS:
        piezas, info, t = [], [], 0.0
        lista = frases(esc["tts"])
        for i, (texto, pausa) in enumerate(lista):
            a, sr = k.create(texto, voice=VOZ, speed=VELOCIDAD, lang=IDIOMA)
            a = recortar(np.asarray(a, dtype=np.float32), sr)
            dur = len(a) / sr
            info.append(dict(texto=texto, inicio=round(t, 3), fin=round(t + dur, 3)))
            piezas.append(a)
            t += dur
            if i < len(lista) - 1:
                piezas.append(np.zeros(int(sr * pausa), dtype=np.float32))
                t += pausa
        audio = np.concatenate(piezas)
        audio = audio / max(1e-6, np.abs(audio).max()) * 0.89  # pico ~ -1 dBFS
        sf.write(os.path.join(salida, f"escena_{esc['id']}.wav"), audio, sr, subtype="FLOAT")
        tiempos[esc["id"]] = dict(duracion=round(len(audio) / sr, 3), frases=info)
        print(f"escena {esc['id']}: {len(audio) / sr:5.2f} s  " + " | ".join(f"{f['inicio']:.2f} {f['texto']}" for f in info))
        sys.stdout.flush()
    with open(os.path.join(salida, "tiempos.json"), "w") as f:
        json.dump(tiempos, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

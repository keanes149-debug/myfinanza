#!/usr/bin/env python3
"""Musica de fondo generada con numpy (118 BPM), efectos de sonido, voz, ducking y loudnorm.

Lee build/plan.json y build/voz/escena_N.wav. Escribe build/audio/mezcla_final.wav (48 kHz estereo).
"""
import functools
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

from guion import BUILD

SR = 48000
BPM = 118
BEAT = 60 / BPM
BAR = 4 * BEAT
rng = np.random.default_rng(7)


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tiempo(n):
    return np.arange(n) / SR


def filtro(x, lp=None, hp=None, orden=2):
    """Filtro de fase cero en el dominio de la frecuencia (Butterworth en magnitud)."""
    n = len(x)
    N = 1 << int(np.ceil(np.log2(n * 2)))
    X = np.fft.rfft(x, N)
    f = np.fft.rfftfreq(N, 1 / SR)
    Hf = np.ones_like(f)
    if lp:
        Hf *= 1 / np.sqrt(1 + (f / lp) ** (2 * orden))
    if hp:
        Hf *= 1 / np.sqrt(1 + (hp / np.maximum(f, 1e-3)) ** (2 * orden))
    return np.fft.irfft(X * Hf, N)[:n]


def remuestrear(x, sr_in, sr_out):
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    return np.fft.irfft(Y, n_out) * (n_out / len(x))


def colocar(pista, x, t, gan=1.0, pan=0.0):
    """Suma x (mono) en la pista estereo en el segundo t con paneo (-1 izq .. 1 der)."""
    i = int(round(t * SR))
    if i >= pista.shape[0]:
        return
    if i < 0:
        x, i = x[-i:], 0
    x = x[: pista.shape[0] - i]
    gl = gan * np.cos((pan + 1) * np.pi / 4) * np.sqrt(2)
    gr = gan * np.sin((pan + 1) * np.pi / 4) * np.sqrt(2)
    pista[i:i + len(x), 0] += x * gl
    pista[i:i + len(x), 1] += x * gr


# ----------------------------------------------------------------- instrumentos
def bombo():
    n = int(0.42 * SR)
    t = tiempo(n)
    f = 46 + 110 * np.exp(-t * 32)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 7.5) + 0.45 * np.sin(2 * ph) * np.exp(-t * 28)
    y += filtro(rng.standard_normal(n), hp=1500) * np.exp(-t * 300) * 0.2
    return y * 0.9


def palmas():
    n = int(0.3 * SR)
    t = tiempo(n)
    env = np.zeros(n)
    for off in (0, 0.011, 0.022):
        tt = t - off
        env += np.where(tt >= 0, np.exp(-np.maximum(tt, 0) * 170), 0)
    env += np.where(t > 0.022, np.exp(-(t - 0.022) * 14) * 0.35, 0)
    return filtro(rng.standard_normal(n) * env, lp=5000, hp=900) * 0.8


def platillo(dur=0.05, caida=70.0, hp=7000):
    n = int(dur * SR)
    t = tiempo(n)
    return filtro(rng.standard_normal(n), hp=hp) * np.exp(-t * caida)


def maraca():
    n = int(0.07 * SR)
    t = tiempo(n)
    env = np.sin(np.pi * t / t[-1]) ** 2
    return filtro(rng.standard_normal(n), hp=5000) * env


@functools.lru_cache(None)
def bajo(m, dur):
    n = int(dur * SR)
    t = tiempo(n)
    f = mtof(m)
    y = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(4 * np.pi * f * t) + 0.25 * np.sin(6 * np.pi * f * t)
         + 0.12 * np.sin(8 * np.pi * f * t))
    env = (1 - np.exp(-t * 300)) * (0.55 + 0.45 * np.exp(-t * 9))
    env *= np.clip((dur - t) / 0.03, 0, 1)
    return y * env


@functools.lru_cache(None)
def pulsado(m, dur=0.45):
    n = int(dur * SR)
    t = tiempo(n)
    f = mtof(m)
    y = sum((1 / k ** 1.3) * np.sin(2 * np.pi * k * f * t + k) * np.exp(-t * (5 + 3.5 * k)) for k in range(1, 7))
    return y * (1 - np.exp(-t * 600))


@functools.lru_cache(None)
def colchon(notas, dur):
    n = int(dur * SR)
    t = tiempo(n)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in notas:
        for det, canal in ((-0.07, L), (0.07, R)):
            f = mtof(m + det)
            canal += sum((1 / k) * np.sin(2 * np.pi * k * f * t + k * 0.7) for k in range(1, 7))
    env = np.clip(t / 0.35, 0, 1) * np.clip((dur - t) / 0.4, 0, 1)
    return L * env, R * env


def campana(f, dur=1.4):
    n = int(dur * SR)
    t = tiempo(n)
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t * 3) + 0.4 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t * 6)
         + 0.2 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-t * 11))
    return y * (1 - np.exp(-t * 900))


# ----------------------------------------------------------------- efectos
def efecto(tipo, idx=0):
    if tipo == "notificacion":
        a = 1 + 0.04 * (idx % 3)
        n1, n2 = int(0.075 * SR), int(0.16 * SR)
        t1, t2 = tiempo(n1), tiempo(n2)
        y1 = np.sin(2 * np.pi * 1568 * a * t1) * np.exp(-t1 * 30)
        y2 = np.sin(2 * np.pi * 2093 * a * t2) * np.exp(-t2 * 18)
        return np.concatenate([y1, y2]) * 0.28, (0.35 if idx % 2 else -0.35)
    if tipo in ("pop", "pop_suave"):
        n = int(0.09 * SR)
        t = tiempo(n)
        f0, f1 = (380, 820) if tipo == "pop" else (620, 1150)
        f = f0 + (f1 - f0) * np.clip(t / 0.05, 0, 1)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 45)
        return y * (0.42 if tipo == "pop" else 0.26), 0.0
    if tipo == "exito":
        y = np.zeros(int(1.0 * SR))
        for k, f in enumerate((1318.5, 1975.5)):
            c = campana(f, 0.9) * 0.22
            i = int(k * 0.075 * SR)
            y[i:i + len(c)] += c[: len(y) - i]
        return y, 0.0
    if tipo == "campana":
        y = np.zeros(int(1.6 * SR))
        for k, f in enumerate((1174.7, 1760.0, 2349.3)):
            c = campana(f, 1.4) * 0.16
            i = int(k * 0.06 * SR)
            y[i:i + len(c)] += c[: len(y) - i]
        return y, 0.0
    if tipo in ("whoosh", "desliza", "sale"):
        dur = {"whoosh": 0.42, "desliza": 0.32, "sale": 0.6}[tipo]
        n = int(dur * SR)
        t = tiempo(n)
        y = filtro(rng.standard_normal(n), lp=6000 if tipo != "sale" else 2500, hp=500 if tipo != "sale" else 250)
        y *= np.sin(np.pi * t / t[-1]) ** 2
        return y * {"whoosh": 0.16, "desliza": 0.12, "sale": 0.14}[tipo], 0.0
    if tipo == "vibra":
        n = int(0.5 * SR)
        t = tiempo(n)
        y = np.sin(2 * np.pi * 165 * t) + 0.5 * np.sign(np.sin(2 * np.pi * 165 * t))
        env = np.clip(t / 0.02, 0, 1) * np.clip((0.5 - t) / 0.05, 0, 1) * (0.7 + 0.3 * np.sin(2 * np.pi * 28 * t))
        return filtro(y * env, lp=1600) * 0.16, 0.0
    if tipo == "tic":
        n = int(0.03 * SR)
        t = tiempo(n)
        return np.sin(2 * np.pi * 1900 * t) * np.exp(-t * 160) * 0.1, 0.0
    if tipo == "toque":
        n = int(0.05 * SR)
        t = tiempo(n)
        y = np.sin(2 * np.pi * 1250 * t) * np.exp(-t * 110) + filtro(rng.standard_normal(n), hp=2500) * np.exp(-t * 400) * 0.4
        return y * 0.3, 0.0
    if tipo == "sello":
        n = int(0.3 * SR)
        t = tiempo(n)
        f = 60 + 70 * np.exp(-t * 25)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14)
        y += filtro(rng.standard_normal(n), lp=1800) * np.exp(-t * 60) * 0.5
        return y * 0.5, 0.0
    raise KeyError(tipo)


# ----------------------------------------------------------------- musica
PROG = [(38, [62, 66, 69]), (45, [61, 64, 69]), (47, [62, 66, 71]), (43, [62, 67, 71])]  # D A Bm G


def musica(total, plan):
    n = int(round(total * SR))
    pista = np.zeros((n + SR, 2))   # bateria (sin sidechain)
    ton = np.zeros((n + SR, 2))     # colchon, acordes y bajo (respiran con el bombo)
    t_solucion = plan["escenas"][2]["inicio"]
    t_problema = plan["escenas"][1]["inicio"]
    ult = plan["escenas"][-1]
    t_outro = ult["voz_inicio"] + (ult["Ff"][-1] - ult["F"][0]) + 0.6
    K, P, HH, OH, SH = bombo(), palmas(), platillo(), platillo(0.2, 16, 6000), maraca()
    nb = int(np.ceil(total / BAR)) + 1
    pump = np.ones(n + SR)
    for b in range(nb):
        t0 = b * BAR
        raiz, acorde = PROG[b % 4]
        fase = "intro" if b == 0 else "outro" if t0 >= t_outro - 0.01 else \
            "quiebre" if t_problema - 0.2 <= t0 < t_solucion - 0.5 else "lleno"
        if fase == "outro":
            continue
        # colchon armonico siempre
        L, R = colchon(tuple(acorde), BAR + 0.3)
        colocar(ton, L * 0.05, t0, pan=-0.8)
        colocar(ton, R * 0.05, t0, pan=0.8)
        for s16 in range(16):
            ts = t0 + s16 * BEAT / 4
            if ts >= total:
                break
            if fase == "lleno" and s16 % 4 == 0:
                colocar(pista, K, ts, 0.95)
                i = int(ts * SR)
                m = min(len(pump) - i, int(BEAT * SR))
                pump[i:i + m] = 1 - 0.45 * np.exp(-tiempo(m) * 9)
            if fase == "lleno" and s16 in (4, 12):
                colocar(pista, P, ts, 0.5, 0.05)
            if s16 % 4 == 2:
                colocar(pista, OH if (fase == "lleno" and s16 == 14) else HH, ts, 0.22 if fase == "lleno" else 0.14, 0.3)
                for k, nota in enumerate(acorde):
                    colocar(ton, pulsado(nota + 12 * (fase == "intro"), 0.4), ts,
                            0.11 if fase != "quiebre" else 0.08, -0.4 if (s16 // 4) % 2 else 0.4)
                if fase == "lleno":
                    colocar(ton, bajo(raiz + (12 if s16 == 14 else 0), BEAT * 0.42), ts, 0.42)
            if fase == "quiebre" and s16 in (0, 8):
                colocar(ton, bajo(raiz, BEAT * 1.6), ts, 0.3)
            if fase != "intro":
                colocar(pista, SH, ts, 0.06 if s16 % 2 else 0.035, -0.3)
    # sidechain: colchon, bajo y acordes "respiran" con el bombo
    pista += ton * pump[:, None]
    # subida (riser) e impacto al entrar la solucion
    dur_r = 1.5
    nr = int(dur_r * SR)
    tr = tiempo(nr)
    riser = filtro(rng.standard_normal(nr), hp=900) * (tr / dur_r) ** 2.2 * 0.22
    colocar(pista, riser, t_solucion - dur_r)
    colocar(pista, K, t_solucion, 1.0)
    choque = platillo(1.6, 2.6, 4500) * 0.35
    colocar(pista, choque, t_solucion, pan=-0.2)
    colocar(pista, choque * 0.9, t_solucion + 0.012, pan=0.2)
    # final: acorde largo + pequena melodia de campana
    L, R = colchon(tuple(PROG[0][1]), round(total - t_outro + 0.5, 3))
    colocar(pista, L * 0.06, t_outro, pan=-0.8)
    colocar(pista, R * 0.06, t_outro, pan=0.8)
    colocar(pista, K, t_outro, 0.8)
    colocar(pista, bajo(38, 1.6), t_outro, 0.4)
    for k, m in enumerate([74, 78, 81, 86]):
        colocar(pista, campana(mtof(m), 1.6) * 0.12, t_outro + k * BEAT / 2, pan=(-0.3 + 0.2 * k))
    pista = pista[:n]
    # desvanecido final
    fade = int(1.2 * SR)
    pista[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5
    return pista


# ----------------------------------------------------------------- voz y ducking
def pista_voz(total, plan):
    n = int(round(total * SR))
    voz = np.zeros(n)
    for s in plan["escenas"]:
        x, sr = sf.read(os.path.join(BUILD, "voz", f"escena_{s['id']}.wav"), dtype="float64")
        x = remuestrear(x, sr, SR)
        i = int(round(s["voz_inicio"] * SR))
        voz[i:i + len(x)] += x[: n - i]
    return voz


def ducking(voz, profundidad_db=-13.0):
    """Ganancia para la musica: baja cuando habla la voz (con anticipacion y relajacion suave)."""
    marco = int(0.01 * SR)
    nf = len(voz) // marco + 1
    v = np.pad(voz, (0, nf * marco - len(voz)))
    rms = np.sqrt((v.reshape(nf, marco) ** 2).mean(1))
    activo = (rms > 0.02).astype(float)
    # une huecos cortos entre palabras/frases (< 0.45 s) para que la musica no "bombee"
    k = 45
    activo = np.convolve(activo, np.ones(k), mode="same") > 0
    activo = activo.astype(float)
    activo = np.roll(activo, -8)          # anticipacion de 80 ms
    g_baja = 10 ** (profundidad_db / 20)
    obj = 1 - (1 - g_baja) * activo
    g = np.ones(nf)
    a_baj, a_sub = 1 - np.exp(-1 / 6), 1 - np.exp(-1 / 35)  # ~60 ms para bajar, ~350 ms para subir
    for i in range(1, nf):
        a = a_baj if obj[i] < g[i - 1] else a_sub
        g[i] = g[i - 1] + a * (obj[i] - g[i - 1])
    return np.interp(np.arange(len(voz)), np.arange(nf) * marco, g)


def loudnorm(entrada, salida, I=-14.0, TP=-1.5, LRA=11.0):
    base = f"loudnorm=I={I}:TP={TP}:LRA={LRA}"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", entrada, "-af", base + ":print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    med = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
    filtro_ln = (f"{base}:measured_I={med['input_i']}:measured_TP={med['input_tp']}:measured_LRA={med['input_lra']}"
                 f":measured_thresh={med['input_thresh']}:offset={med['target_offset']}:linear=true:print_format=json")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", entrada, "-af", filtro_ln, "-ar", str(SR),
                        "-c:a", "pcm_s24le", salida], capture_output=True, text=True)
    final = json.loads(re.findall(r"\{[^{}]*\}", r.stderr)[-1])
    return med, final


def main():
    with open(os.path.join(BUILD, "plan.json")) as f:
        plan = json.load(f)
    total = plan["total"]
    d = os.path.join(BUILD, "audio")
    os.makedirs(d, exist_ok=True)
    voz = pista_voz(total, plan)
    mus = musica(total, plan)
    # nivel: musica (sin voz) ~4 dB por debajo de la voz; con voz baja 13 dB mas (ducking)
    rms_voz = np.sqrt(np.mean(voz[np.abs(voz) > 0.02] ** 2))
    rms_mus = np.sqrt(np.mean(mus ** 2))
    mus *= (rms_voz * 10 ** (-4 / 20)) / rms_mus
    mus *= ducking(voz)[:, None]
    sfx = np.zeros_like(mus)
    cuenta = {}
    for t, tipo in plan["eventos"]:
        k = cuenta.get(tipo, 0)
        cuenta[tipo] = k + 1
        x, pan = efecto(tipo, k)
        colocar(sfx, x, t, 0.8, pan)
    mezcla = mus + sfx + voz[:, None]
    pico = np.abs(mezcla).max()
    if pico > 0.98:
        mezcla *= 0.98 / pico
    sf.write(os.path.join(d, "mezcla.wav"), mezcla.astype(np.float32), SR, subtype="FLOAT")
    sf.write(os.path.join(d, "solo_musica.wav"), mus.astype(np.float32), SR, subtype="FLOAT")
    med, final = loudnorm(os.path.join(d, "mezcla.wav"), os.path.join(d, "mezcla_final.wav"))
    print(f"musica+efectos+voz: {total:.2f} s | antes {med['input_i']} LUFS, pico {med['input_tp']} dBTP"
          f" -> despues {final['output_i']} LUFS, pico {final['output_tp']} dBTP ({final['normalization_type']})")
    print("efectos:", cuenta)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dibuja las 8 escenas del video con Pillow y codifica cada una con ffmpeg.

Uso:
  python3 escenas.py plan            calcula la duracion de cada escena a partir de la voz
  python3 escenas.py fotos           guarda cuadros de revision en build/fotos/
  python3 escenas.py render [ids]    renderiza build/escenas/escena_N.mp4 (en paralelo)
"""
import functools
import json
import math
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from guion import (BLANCO, BUILD, CREMA, ESCENAS, FPS, FUENTE, GRIS, GRIS_CLARO, H, ROJO, TEXTO,
                   TRANSICION, VERDE, VERDE_CLARO, VERDE_MEDIO, VERDE_OSCURO, VOZ_RETARDO, W)

SS = 4                        # supersampling para formas suaves
RS = Image.Resampling
FONDO_CHAT = (0xEC, 0xE8, 0xDE)
BEIGE = (0xF3, 0xF0, 0xE8)
ROJO_CLARO = (0xF8, 0xE1, 0xDD)
LINEA = (0xE4, 0xE0, 0xD5)
SOMBRA = (18, 42, 30)


# ----------------------------------------------------------------- utilidades
def cl(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def eo(x):  # ease-out cubico
    x = cl(x)
    return 1 - (1 - x) ** 3


def eio(x):  # ease-in-out cubico
    x = cl(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def eback(x, s=1.9):
    x = cl(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def lerp(a, b, x):
    return a + (b - a) * x


def aparecer(t, t0, d=0.3):
    return cl((t - t0) / d)


def pop(t, t0, d=0.32):
    """Escala de una animacion 'pop': 0 antes de t0, crece con un pequeno rebote hasta 1."""
    x = (t - t0) / d
    if x <= 0:
        return 0.0
    return 0.55 + 0.45 * eback(x)


def flotar(t, amp=6.0, per=3.6, fase=0.0):
    return amp * math.sin(2 * math.pi * t / per + fase)


# ----------------------------------------------------------------- fuentes y sprites
@functools.lru_cache(None)
def fuente(size, peso=400):
    if os.path.exists(FUENTE):
        f = ImageFont.truetype(FUENTE, size)
        try:
            f.set_variation_by_axes([max(9, min(40, size / 2.6)), peso])
        except Exception:
            pass
        return f
    nombre = {300: "Light", 400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold"}
    return ImageFont.truetype(f"/usr/share/fonts/opentype/inter/Inter-{nombre.get(peso, 'Regular')}.otf", size)


class Sp:
    """Imagen RGBA con un origen logico (ox, oy) y tamano logico (w, h) sin contar la sombra."""
    __slots__ = ("img", "ox", "oy", "w", "h", "vc")

    def __init__(self, img, ox=0, oy=0, w=None, h=None, vc=None):
        self.img, self.ox, self.oy = img, ox, oy
        self.w = img.width if w is None else w
        self.h = img.height if h is None else h
        self.vc = self.h / 2 if vc is None else vc


def colorear(mask, color):
    im = Image.new("RGBA", mask.size, tuple(color) + (0,))
    im.putalpha(mask)
    return im


def alfa(img, a):
    if a >= 0.997:
        return img
    im = img.copy()
    im.putalpha(img.getchannel("A").point(lambda v: int(v * a)))
    return im


def pegar(c, sp, x, y, s=1.0, a=1.0, ancla=None):
    """Pega el sprite con su origen logico en (x, y), escalado alrededor de 'ancla' (coords logicas)."""
    if a <= 0.004 or s <= 0.02:
        return
    img = sp.img
    ax, ay = ancla if ancla else (sp.w / 2, sp.h / 2)
    if abs(s - 1) > 0.002:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), RS.BICUBIC)
    img = alfa(img, a)
    px = x + ax - (ax + sp.ox) * s
    py = y + ay - (ay + sp.oy) * s
    c.paste(img, (round(px), round(py)), img)


def lienzo(w, h):
    m = Image.new("L", (max(1, int(w * SS)), max(1, int(h * SS))), 0)
    return m, ImageDraw.Draw(m)


def red(m):
    return m.reduce(SS)


@functools.lru_cache(None)
def m_rrect(w, h, r, esquinas=(True, True, True, True)):
    m, d = lienzo(w, h)
    d.rounded_rectangle([0, 0, w * SS - 1, h * SS - 1], r * SS, fill=255, corners=esquinas)
    return red(m)


@functools.lru_cache(None)
def m_borde(w, h, r, b):
    ext = m_rrect(w, h, r)
    inn = Image.new("L", (w, h), 0)
    inn.paste(m_rrect(w - 2 * b, h - 2 * b, max(1, r - b)), (b, b))
    return ImageChops.subtract(ext, inn)


@functools.lru_cache(None)
def m_disco(d):
    m, dr = lienzo(d, d)
    dr.ellipse([0, 0, d * SS - 1, d * SS - 1], fill=255)
    return red(m)


@functools.lru_cache(None)
def m_anillo(d, g):
    m, dr = lienzo(d, d)
    dr.ellipse([0, 0, d * SS - 1, d * SS - 1], fill=255)
    dr.ellipse([g * SS, g * SS, (d - g) * SS - 1, (d - g) * SS - 1], fill=0)
    return red(m)


def con_sombra(mask, color, radio=22, dy=12, opac=0.16):
    P = int(radio * 2 + abs(dy))
    w, h = mask.size
    base = Image.new("RGBA", (w + 2 * P, h + 2 * P), SOMBRA + (0,))
    if opac > 0:
        sm = Image.new("L", base.size, 0)
        sm.paste(mask, (P, P + dy))
        sm = sm.filter(ImageFilter.GaussianBlur(radio)).point(lambda v: int(v * opac))
        base = colorear(sm, SOMBRA)
    base.alpha_composite(colorear(mask, color), (P, P))
    return Sp(base, P, P, w, h)


@functools.lru_cache(None)
def sp_tarjeta(w, h, r=44, color=BLANCO, opac=0.13):
    return con_sombra(m_rrect(w, h, r), color, radio=26, dy=14, opac=opac)


@functools.lru_cache(None)
def sp_circulo(d, color, a=1.0, sombra=0.0):
    m = m_disco(d)
    if a < 1:
        m = m.point(lambda v: int(v * a))
    return con_sombra(m, color, radio=14, dy=8, opac=sombra) if sombra else Sp(colorear(m, color))


@functools.lru_cache(None)
def sp_anillo(d, g, color, a=1.0):
    m = m_anillo(d, g)
    if a < 1:
        m = m.point(lambda v: int(v * a))
    return Sp(colorear(m, color))


# --- texto
@functools.lru_cache(None)
def sp_texto(s, size, peso, color):
    f = fuente(size, peso)
    asc, desc = f.getmetrics()
    w = math.ceil(f.getlength(s))
    m = Image.new("L", (w + 8, asc + desc + 8), 0)
    ImageDraw.Draw(m).text((4, 4), s, font=f, fill=255, anchor="la")
    top, bot = f.getbbox("H", anchor="la")[1::2]
    return Sp(colorear(m, color), 4, 4, w, asc + desc, vc=(top + bot) / 2)


def texto(c, s, x, y, size, peso=400, color=TEXTO, al="l", va="t", a=1.0, esc=1.0):
    sp = sp_texto(s, size, peso, color)
    xx = x - (sp.w / 2 if al == "c" else sp.w if al == "r" else 0)
    yy = y - (sp.vc if va == "m" else 0)
    pegar(c, sp, xx, yy, esc, a, ancla=(sp.w / 2, sp.vc))
    return sp.w


def ancho(s, size, peso=400):
    return fuente(size, peso).getlength(s)


def ajustar(s, size, peso, maximo, minimo=56):
    while size > minimo and ancho(s, size, peso) > maximo:
        size -= 1
    return size


def recortar_txt(s, size, peso, maximo):
    if ancho(s, size, peso) <= maximo:
        return s
    while s and ancho(s + "…", size, peso) > maximo:
        s = s[:-1]
    return s.rstrip() + "…"


# ----------------------------------------------------------------- iconos (mascaras)
def _icono(s, dibujar):
    m, d = lienzo(s, s)
    k = s * SS

    def P(u, v):
        return (u * k, v * k)

    dibujar(d, k, P)
    return red(m)


def _linea(d, k, pts, g, fill=255):
    d.line(pts, fill=fill, width=int(g * k), joint="curve")
    for x, y in (pts[0], pts[-1]):
        r = g * k / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


@functools.lru_cache(None)
def mi(nombre, s, p=0):
    """Mascaras de iconos genericos dibujados a mano (nada descargado)."""
    def bocadillo(d, k, P, puntos=True):
        d.rounded_rectangle([*P(.12, .16), *P(.88, .70)], .17 * k, fill=255)
        d.polygon([P(.26, .62), P(.18, .88), P(.46, .66)], fill=255)
        if puntos:
            for u in (.33, .5, .67):
                d.ellipse([*P(u - .055, .43 - .055), *P(u + .055, .43 + .055)], fill=0)

    def check(d, k, P, fill=255, g=.11):
        _linea(d, k, [P(.28, .52), P(.44, .67), P(.73, .36)], g, fill)

    if nombre == "chat":
        return _icono(s, bocadillo)
    if nombre == "check":
        return _icono(s, lambda d, k, P: check(d, k, P))
    if nombre == "check_circulo":
        def f(d, k, P):
            d.ellipse([*P(.04, .04), *P(.96, .96)], fill=255)
            check(d, k, P, fill=0, g=.12)
        return _icono(s, f)
    if nombre == "prohibido":
        def f(d, k, P):
            d.ellipse([*P(.04, .04), *P(.96, .96)], fill=255)
            d.rounded_rectangle([*P(.22, .42), *P(.78, .58)], .04 * k, fill=0)
        return _icono(s, f)
    if nombre == "calendario":
        def f(d, k, P):
            d.rounded_rectangle([*P(.1, .18), *P(.9, .9)], .13 * k, fill=255)
            d.rounded_rectangle([*P(.18, .4), *P(.82, .82)], .06 * k, fill=0)
            for u in (.32, .68):
                d.rounded_rectangle([*P(u - .05, .06), *P(u + .05, .3)], .05 * k, fill=255)
            for u in (.33, .5, .67):
                for v in (.53, .69):
                    d.ellipse([*P(u - .045, v - .045), *P(u + .045, v + .045)], fill=255)
        return _icono(s, f)
    if nombre == "reloj":
        minutos = p
        def f(d, k, P):
            d.ellipse([*P(.05, .05), *P(.95, .95)], fill=255)
            d.ellipse([*P(.13, .13), *P(.87, .87)], fill=0)
            for i in range(12):
                a = i / 12 * 2 * math.pi
                r0, r1 = (.27, .33) if i % 3 else (.25, .33)
                _linea(d, k, [P(.5 + r0 * math.cos(a), .5 + r0 * math.sin(a)),
                              P(.5 + r1 * math.cos(a), .5 + r1 * math.sin(a))], .035)
            ah = (minutos % 720) / 720 * 2 * math.pi - math.pi / 2
            am = (minutos % 60) / 60 * 2 * math.pi - math.pi / 2
            _linea(d, k, [P(.5, .5), P(.5 + .17 * math.cos(ah), .5 + .17 * math.sin(ah))], .065)
            _linea(d, k, [P(.5, .5), P(.5 + .26 * math.cos(am), .5 + .26 * math.sin(am))], .045)
            d.ellipse([*P(.455, .455), *P(.545, .545)], fill=255)
        return _icono(s, f)
    if nombre == "luna":
        def f(d, k, P):
            d.ellipse([*P(.08, .08), *P(.92, .92)], fill=255)
            d.ellipse([*P(.3, -.06), *P(1.06, .7)], fill=0)
        return _icono(s, f)
    if nombre == "estrella":
        def f(d, k, P):
            pts = []
            for i in range(8):
                a = i / 8 * 2 * math.pi - math.pi / 2
                r = .48 if i % 2 == 0 else .14
                pts.append(P(.5 + r * math.cos(a), .5 + r * math.sin(a)))
            d.polygon(pts, fill=255)
        return _icono(s, f)
    if nombre == "persona":
        def f(d, k, P):
            d.ellipse([*P(.32, .08), *P(.68, .44)], fill=255)
            d.rounded_rectangle([*P(.16, .5), *P(.84, .98)], .26 * k, fill=255, corners=(True, True, False, False))
        return _icono(s, f)
    if nombre == "lupa":
        def f(d, k, P):
            d.ellipse([*P(.1, .1), *P(.68, .68)], fill=255)
            d.ellipse([*P(.19, .19), *P(.59, .59)], fill=0)
            _linea(d, k, [P(.62, .62), P(.86, .86)], .13)
        return _icono(s, f)
    if nombre == "lapiz":
        def f(d, k, P):
            a, b = P(.78, .14), P(.9, .26)
            d.polygon([P(.7, .2), P(.82, .32), P(.34, .8), P(.22, .68)], fill=255)
            d.polygon([P(.22, .68), P(.34, .8), P(.12, .9)], fill=255)
            d.polygon([P(.74, .16), P(.8, .1), P(.92, .22), P(.86, .28)], fill=255)
        return _icono(s, f)
    if nombre == "enviar":
        return _icono(s, lambda d, k, P: d.polygon([P(.18, .18), P(.86, .5), P(.18, .82), P(.28, .5)], fill=255))
    if nombre == "flecha":
        def f(d, k, P):
            _linea(d, k, [P(.12, .5), P(.8, .5)], .12)
            _linea(d, k, [P(.56, .26), P(.82, .5), P(.56, .74)], .12)
        return _icono(s, f)
    if nombre == "tijeras":
        ap = p
        def f(d, k, P):
            cx, cy, giro = .52, .44, -32
            for sg in (-1, 1):
                ang = math.radians(-90 + giro + sg * ap / 2)
                dx, dy = math.cos(ang), math.sin(ang)
                nx, ny = -dy, dx

                def pt(a, b):
                    return P(cx + dx * a + nx * b, cy + dy * a + ny * b)
                d.polygon([pt(-.02, .05 * sg), pt(.2, .045 * sg), pt(.44, 0), pt(.2, -.012 * sg), pt(0, -.04 * sg)], fill=255)
                hx, hy = cx - dx * .27 - nx * .07 * sg, cy - dy * .27 - ny * .07 * sg
                _linea(d, k, [P(cx, cy), P(hx, hy)], .07)
                d.ellipse([*P(hx - .13, hy - .13), *P(hx + .13, hy + .13)], fill=255)
                d.ellipse([*P(hx - .075, hy - .075), *P(hx + .075, hy + .075)], fill=0)
            d.ellipse([*P(cx - .05, cy - .05), *P(cx + .05, cy + .05)], fill=255)
            d.ellipse([*P(cx - .02, cy - .02), *P(cx + .02, cy + .02)], fill=0)
        return _icono(s, f)
    if nombre == "ondas":
        def f(d, k, P):
            for r, g in ((.3, .045), (.44, .045)):
                d.arc([*P(.5 - r, .5 - r), *P(.5 + r, .5 + r)], 150, 210, fill=255, width=int(g * k))
                d.arc([*P(.5 - r, .5 - r), *P(.5 + r, .5 + r)], -30, 30, fill=255, width=int(g * k))
        return _icono(s, f)
    raise KeyError(nombre)


@functools.lru_cache(None)
def sp_icono(nombre, s, color, p=0):
    return Sp(colorear(mi(nombre, s, p), color))


@functools.lru_cache(None)
def sp_emoji(nombre, s):
    """Reemplazos dibujados de los emojis ✅ y ⛔ (para no depender de fuentes de emoji)."""
    if nombre == "✅":
        base = colorear(m_rrect(s, s, int(s * .24)), VERDE_MEDIO)
        base.alpha_composite(colorear(mi("check", s), BLANCO))
    else:
        base = colorear(m_disco(s), ROJO)
        m, d = lienzo(s, s)
        d.rounded_rectangle([.2 * s * SS, .41 * s * SS, .8 * s * SS, .59 * s * SS], .05 * s * SS, fill=255)
        base.alpha_composite(colorear(red(m), BLANCO))
    return Sp(base)


@functools.lru_cache(None)
def sp_logo(s, fondo, figura, sombra=0.18):
    """Marca generica: cuadro redondeado + globo de chat + check."""
    sp = con_sombra(m_rrect(s, s, int(s * .28)), fondo, radio=int(s * .08), dy=int(s * .05), opac=sombra)
    img = sp.img.copy()
    g = int(s * .74)
    off = (sp.ox + (s - g) // 2, sp.oy + int((s - g) / 2) + int(s * .03))
    img.alpha_composite(colorear(mi("chat", g).point(lambda v: v), figura), off)
    # tapamos los puntos del globo con un check
    m = Image.new("L", (g, g), 0)
    dr = ImageDraw.Draw(m)
    dr.rounded_rectangle([.2 * g, .22 * g, .8 * g, .64 * g], .1 * g, fill=255)
    img.alpha_composite(colorear(m, figura), off)
    img.alpha_composite(colorear(mi("check", int(g * .78)), fondo), (off[0] + int(g * .11), off[1] + int(g * .0)))
    return Sp(img, sp.ox, sp.oy, s, s)


@functools.lru_cache(None)
def sp_icono_app(s):
    base = colorear(m_rrect(s, s, int(s * .28)), VERDE_CLARO)
    base.alpha_composite(colorear(mi("chat", int(s * .78)), VERDE), (int(s * .11), int(s * .13)))
    return Sp(base)


@functools.lru_cache(None)
def sp_avatar(s, icono, fondo=VERDE_CLARO, color=VERDE):
    base = colorear(m_disco(s), fondo)
    g = int(s * .62)
    base.alpha_composite(colorear(mi(icono, g, 22 if icono == "tijeras" else 0), color), ((s - g) // 2, (s - g) // 2))
    return Sp(base)


@functools.lru_cache(None)
def sp_avatar_iniciales(s, nombre):
    base = colorear(m_disco(s), VERDE_CLARO)
    ini = "".join(p[0] for p in nombre.split()[:2]).upper()
    sp = sp_texto(ini, 40, 700, VERDE)
    base.alpha_composite(sp.img, (int(s / 2 - sp.w / 2 - sp.ox), int(s / 2 - sp.vc - sp.oy)))
    return Sp(base)


@functools.lru_cache(None)
def sp_pastilla(txt, size, peso, fondo, color, icono=None, h=None, sombra=0.10, padx=None):
    f = fuente(size, peso)
    tw = f.getlength(txt)
    h = h or int(size * 1.8)
    padx = padx or int(size * 0.7)
    isz = int(size * 1.05) if icono else 0
    gap = int(size * 0.35) if icono else 0
    w = int(tw + 2 * padx + isz + gap)
    sp = con_sombra(m_rrect(w, h, h // 2), fondo, radio=12, dy=6, opac=sombra)
    img = sp.img.copy()
    x = sp.ox + padx
    if icono:
        if icono in ("✅", "⛔"):
            img.alpha_composite(sp_emoji(icono, isz).img, (x, sp.oy + (h - isz) // 2))
        else:
            img.alpha_composite(colorear(mi(icono, isz), color), (x, sp.oy + (h - isz) // 2))
        x += isz + gap
    t = sp_texto(txt, size, peso, color)
    img.alpha_composite(t.img, (int(x - t.ox), int(sp.oy + h / 2 - t.vc - t.oy)))
    return Sp(img, sp.ox, sp.oy, w, h)


# ----------------------------------------------------------------- burbujas de chat
FS_CHAT, LH_CHAT = 42, 57
PADX_B, PADY_B = 30, 22
TIPEO_W, TIPEO_H = 156, 98


def envolver(s, size, peso, maximo):
    f = fuente(size, peso)
    esp = f.getlength(" ")
    lineas, actual = [[]], 0.0
    for tok in s.split(" "):
        w = size * 1.02 if tok in ("✅", "⛔") else f.getlength(tok)
        if lineas[-1] and actual + esp + w > maximo:
            lineas.append([])
            actual = 0.0
        actual += (esp if lineas[-1] else 0) + w
        lineas[-1].append((tok, w))
    return lineas, esp


@functools.lru_cache(None)
def sp_burbuja(txt, lado, fondo, maximo=800):
    lineas, esp = envolver(txt, FS_CHAT, 400, maximo - 2 * PADX_B)
    anchos = [sum(w for _, w in l) + esp * (len(l) - 1) for l in lineas]
    w = int(max(anchos) + 2 * PADX_B)
    h = int(len(lineas) * LH_CHAT + 2 * PADY_B)
    esquinas = (False, True, True, True) if lado == "izq" else (True, False, True, True)
    sp = con_sombra(m_rrect(w, h, 30, esquinas), fondo, radio=8, dy=4, opac=0.10)
    img = sp.img.copy()
    f = fuente(FS_CHAT, 400)
    tm = Image.new("L", img.size, 0)
    dr = ImageDraw.Draw(tm)
    iconos = []
    for i, l in enumerate(lineas):
        x = sp.ox + PADX_B
        y = sp.oy + PADY_B + i * LH_CHAT + 3
        for tok, tw in l:
            if tok in ("✅", "⛔"):
                isz = int(FS_CHAT * 1.02)
                iconos.append((tok, isz, int(x), int(y + 4)))
            else:
                dr.text((x, y), tok, font=f, fill=255, anchor="la")
            x += tw + esp
    img.alpha_composite(colorear(tm, TEXTO))
    for tok, isz, x, y in iconos:
        img.alpha_composite(sp_emoji(tok, isz).img, (x, y))
    return Sp(img, sp.ox, sp.oy, w, h)


@functools.lru_cache(None)
def sp_tipeo():
    return con_sombra(m_rrect(TIPEO_W, TIPEO_H, 30, (False, True, True, True)), BLANCO, radio=8, dy=4, opac=0.10)


@functools.lru_cache(None)
def sp_degradado(w, h, color):
    g = Image.linear_gradient("L").resize((w, h)).point(lambda v: 255 - v)
    return Sp(colorear(g, color))


HEAD_H, INPUT_H = 150, 104


@functools.lru_cache(None)
def sp_chat_base(w, h, nombre, icono):
    sp = sp_tarjeta(w, h, 44)
    img = sp.img.copy()
    o = (sp.ox, sp.oy)
    img.alpha_composite(colorear(m_rrect(w, HEAD_H, 44, (True, True, False, False)), VERDE), o)
    av = sp_avatar(92, icono)
    img.alpha_composite(av.img, (o[0] + 30, o[1] + (HEAD_H - 92) // 2))
    t = sp_texto(nombre, 46, 700, BLANCO)
    img.alpha_composite(t.img, (o[0] + 146 - t.ox, o[1] + 22 - t.oy))
    # barra de escritura
    yb = o[1] + h - INPUT_H
    campo = colorear(m_rrect(w - 170, 72, 36), BEIGE)
    img.alpha_composite(campo, (o[0] + 28, yb + 16))
    t = sp_texto("Escribe un mensaje", 40, 400, GRIS)
    img.alpha_composite(t.img, (o[0] + 62 - t.ox, int(yb + 16 + 36 - t.vc - t.oy)))
    env = colorear(m_disco(76), VERDE)
    env.alpha_composite(colorear(mi("enviar", 44), BLANCO), (18, 16))
    img.alpha_composite(env, (o[0] + w - 28 - 76 - 8, yb + 14))
    return Sp(img, sp.ox, sp.oy, w, h)


class Chat:
    """Tarjeta de chat con burbujas que aparecen con 'pop' y un indicador 'escribiendo…'."""

    def __init__(self, x, y, w, h, nombre, icono, mensajes, tipeo=0.5):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.nombre, self.icono, self.tipeo = nombre, icono, tipeo
        self.body_h = h - HEAD_H - INPUT_H
        self.msgs = []
        for m in mensajes:
            fondo = BLANCO if m["lado"] == "izq" else VERDE_CLARO
            self.msgs.append(dict(m, sp=sp_burbuja(m["texto"], m["lado"], fondo)))

    def escribiendo(self, t):
        return any(m.get("tip") and m["t"] - self.tipeo <= t < m["t"] for m in self.msgs)

    def dibujar(self, c, t, dy=0.0):
        pegar(c, sp_chat_base(self.w, self.h, self.nombre, self.icono), self.x, self.y + dy)
        estado = "escribiendo…" if self.escribiendo(t) else "en línea"
        texto(c, estado, self.x + 146, self.y + dy + 84, 40, 400, VERDE_CLARO)
        cuerpo = Image.new("RGB", (self.w, self.body_h), FONDO_CHAT)
        items = []
        for m in self.msgs:
            t0 = m["t"]
            if m.get("tip") and t0 - self.tipeo <= t < t0:
                loc = t - (t0 - self.tipeo)
                items.append(("tip", m, TIPEO_H * eo(loc / 0.18), loc))
            elif t >= t0:
                loc = t - t0
                h0 = TIPEO_H if m.get("tip") else 0
                items.append(("msg", m, lerp(h0, m["sp"].h, eo(loc / 0.28)), loc))
        GAP, PADT = 18, 26
        y, pos = PADT, []
        for it in items:
            pos.append(y)
            y += it[2] + GAP
        total = y - GAP + PADT
        scroll = max(0.0, total - self.body_h)
        for it, py in zip(items, pos):
            yy = py - scroll
            tipo, m, _, loc = it
            if tipo == "tip":
                sp = sp_tipeo()
                s = pop(loc, 0, 0.22)
                pegar(cuerpo, sp, 28, yy, s, cl(loc / 0.1), ancla=(0, 0))
                if s > 0.9:
                    for i in range(3):
                        rebote = max(0.0, math.sin(2 * math.pi * (loc * 2.4) - i * 0.9)) * 9
                        pegar(cuerpo, sp_circulo(18, GRIS), 28 + 42 + i * 30 - 9, yy + TIPEO_H / 2 - 9 - rebote)
            else:
                sp = m["sp"]
                izq = m["lado"] == "izq"
                xx = 28 if izq else self.w - 28 - sp.w
                s = lerp(0.82, 1.0, eback(loc / 0.3)) if m.get("tip") else pop(loc, 0, 0.3)
                oculto = cl(-yy / (0.5 * sp.h)) if yy < 0 else 0.0
                pegar(cuerpo, sp, xx, yy, s, cl(loc / 0.12) * (1 - oculto), ancla=(0 if izq else sp.w, 0))
        pegar(cuerpo, sp_degradado(self.w, 44, FONDO_CHAT), 0, 0)
        c.paste(cuerpo, (round(self.x), round(self.y + dy + HEAD_H)))


# ----------------------------------------------------------------- telefono y fondo
BISEL, R_TEL = 20, 96


@functools.lru_cache(None)
def sp_telefono(w, h):
    return con_sombra(m_rrect(w, h, R_TEL), TEXTO, radio=32, dy=20, opac=0.24)


def dibujar_telefono(c, x, y, w, h, pantalla):
    pegar(c, sp_telefono(w, h), x, y)
    m = m_rrect(w - 2 * BISEL, h - 2 * BISEL, R_TEL - BISEL)
    c.paste(pantalla, (round(x + BISEL), round(y + BISEL)), m)
    pegar(c, sp_notch(), x + w / 2 - 95, y + BISEL + 16)


@functools.lru_cache(None)
def sp_notch():
    return Sp(colorear(m_rrect(190, 40, 20), TEXTO))


@functools.lru_cache(None)
def degradado_v(w, h, c0, c1):
    y = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array(c0)[None, None, :] * (1 - y) + np.array(c1)[None, None, :] * y)
    arr = np.repeat(arr, w, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


@functools.lru_cache(None)
def base_color(color):
    return Image.new("RGB", (W, H), color)


MANCHAS_CLARO = [  # cx, cy, d, ampx, ampy, periodo, fase, tipo
    (980, 230, 760, 40, 30, 9.0, 0.0, "disco"),
    (80, 1720, 640, 36, 40, 11.0, 1.3, "disco"),
    (100, 500, 300, 26, 30, 7.5, 2.1, "anillo"),
    (990, 1600, 200, 22, 30, 6.5, 0.7, "anillo"),
]


def fondo(t, oscuro=False):
    c = base_color(VERDE if oscuro else CREMA).copy()
    for cx, cy, d, ax, ay, per, fa, tipo in MANCHAS_CLARO:
        x = cx + ax * math.sin(2 * math.pi * t / per + fa)
        y = cy + ay * math.cos(2 * math.pi * t / (per * 1.3) + fa)
        if oscuro:
            sp = sp_circulo(d, VERDE_MEDIO, 0.28) if tipo == "disco" else sp_anillo(d, 22, VERDE_MEDIO, 0.4)
        else:
            sp = sp_circulo(d, VERDE_CLARO, 0.75) if tipo == "disco" else sp_anillo(d, 22, VERDE_CLARO, 1.0)
        pegar(c, sp, x - d / 2, y - d / 2)
    return c


def encabezado(c, t, paso, titulo, t0=0.18):
    a = aparecer(t, t0, 0.35)
    dy = (1 - eo(a)) * 34
    sp = sp_pastilla(paso, 40, 700, VERDE, BLANCO, sombra=0.0, h=70, padx=34)
    pegar(c, sp, W / 2 - sp.w / 2, 92 + dy, a=a)
    a2 = aparecer(t, t0 + 0.08, 0.35)
    size = ajustar(titulo, 68, 700, 960)
    texto(c, titulo, W / 2, 245 + (1 - eo(a2)) * 34, size, 700, TEXTO, al="c", va="m", a=a2)


# ----------------------------------------------------------------- escena 1: gancho
NOTIS = [("Mariela", "Hola, ¿atienden hoy?"), ("Jhonny", "¿Cuánto cuesta el corte?"),
         ("Daniela", "¿Hay espacio a las 7?"), ("Carlos", "Buenas, ¿tienen turno mañana?"),
         ("Rosa", "¿Hacen uñas acrílicas?"), ("Luis", "¡Hola! ¿Me pueden agendar?"),
         ("Paola", "¿Hasta qué hora atienden?"), ("Diego", "¿Siguen abiertos?"),
         ("Andrea", "¿Hola? ¿Alguien?"), ("Wilson", "Quiero corte y barba"),
         ("Karen", "¿Tienen espacio el sábado?"), ("Marcelo", "¿Atienden el domingo?"),
         ("Fabiola", "Buenas noches, ¿hay turno?"), ("Ronald", "¿Me confirman, por favor?")]
CONTEOS = [1, 2, 3, 5, 6, 8, 10, 12, 15, 17, 20, 23, 26, 31]
NOTI_W, NOTI_H = 668, 142
TEL1 = (160, 380, 760, 1490)


@functools.lru_cache(None)
def sp_noti(j):
    nombre, msg = NOTIS[j]
    m = m_rrect(NOTI_W, NOTI_H, 34)
    sp = con_sombra(m, BLANCO, radio=14, dy=8, opac=0.22)
    img = sp.img.copy()
    o = (sp.ox, sp.oy)
    img.alpha_composite(sp_icono_app(86).img, (o[0] + 24, o[1] + (NOTI_H - 86) // 2))
    t = sp_texto(nombre, 40, 700, TEXTO)
    img.alpha_composite(t.img, (o[0] + 130 - t.ox, o[1] + 20 - t.oy))
    msg = recortar_txt(msg, 40, 400, NOTI_W - 130 - 26)
    t = sp_texto(msg, 40, 400, (0x45, 0x50, 0x4A))
    img.alpha_composite(t.img, (o[0] + 130 - t.ox, o[1] + 74 - t.oy))
    return Sp(img, sp.ox, sp.oy, NOTI_W, NOTI_H)


@functools.lru_cache(None)
def sp_insignia(n):
    sp = con_sombra(m_disco(132), ROJO, radio=12, dy=8, opac=0.25)
    img = sp.img.copy()
    t = sp_texto(str(n), 62, 800, BLANCO)
    img.alpha_composite(t.img, (int(sp.ox + 66 - t.w / 2 - t.ox), int(sp.oy + 66 - t.vc - t.oy)))
    return Sp(img, sp.ox, sp.oy, 132, 132)


def escena1(t, ctx):
    c = fondo(t)
    T = ctx["T"]
    # titulo, palabra por palabra al ritmo de la voz
    lineas = [["¿Cuántos", "clientes", "pierdes"], ["por", "no", "contestar", "a", "tiempo?"]]
    size = min(ajustar(" ".join(l), 74, 800, 980) for l in lineas)
    k = 0
    for li, l in enumerate(lineas):
        esp = ancho(" ", size, 800)
        tot = sum(ancho(p, size, 800) for p in l) + esp * (len(l) - 1)
        x = W / 2 - tot / 2
        for p in l:
            tw = T["palabras"][k]
            a = aparecer(t, tw - 0.04, 0.22)
            col = ROJO if p == "pierdes" else TEXTO
            texto(c, p, x, 128 + li * 100 + (1 - eo(a)) * 26, size, 800, col, a=a)
            x += ancho(p, size, 800) + esp
            k += 1
    # telefono con notificaciones acumulandose
    x, y, w, h = TEL1
    lleg = T["llegadas"]
    vib = sum(9 * math.sin(2 * math.pi * 26 * (t - ta)) * math.exp(-(t - ta) * 10) for ta in lleg if t >= ta)
    x += vib
    y += flotar(t, 5, 3.4)
    sw, sh = w - 2 * BISEL, h - 2 * BISEL
    pant = degradado_v(sw, sh, VERDE_OSCURO, VERDE).copy()
    texto(pant, "19:42", sw / 2, 96, 150, 300, BLANCO, al="c")
    texto(pant, "Viernes 20 de noviembre", sw / 2, 292, 40, 500, VERDE_CLARO, al="c")
    for j, ta in enumerate(lleg):
        if t < ta:
            continue
        slot = sum(eo((t - tb) / 0.25) for tb in lleg[j + 1:] if t >= tb)
        yy = 380 + slot * (NOTI_H + 16)
        if yy > sh + 20:
            continue
        loc = t - ta
        yy -= (1 - eo(loc / 0.25)) * 40
        pegar(pant, sp_noti(j), (sw - NOTI_W) / 2, yy, pop(loc, 0, 0.3), cl(loc / 0.12), ancla=(NOTI_W / 2, 0))
    dibujar_telefono(c, x, y, w, h, pant)
    n = sum(t >= ta for ta in lleg)
    if n:
        ult = max(ta for ta in lleg if t >= ta)
        s = pop(t, lleg[0], 0.3) * (1 + 0.16 * math.exp(-(t - ult) * 9))
        sp = sp_insignia(CONTEOS[n - 1])
        pegar(c, sp, x + w - 96, y - 30, s)
    return c


# ----------------------------------------------------------------- escena 2: el problema
PANELES = [(70, 110, 940, 500), (70, 640, 940, 500), (70, 1170, 940, 650)]


@functools.lru_cache(None)
def sp_movil_chico(w, h):
    img = colorear(m_rrect(w, h, 34), TEXTO)
    img.alpha_composite(colorear(m_rrect(w - 24, h - 24, 24), VERDE_CLARO), (12, 12))
    img.alpha_composite(colorear(mi("chat", int(w * .6)), VERDE), (int(w * .2), int(h * .34)))
    return Sp(img)


@functools.lru_cache(None)
def sp_reloj_grande(d, minutos):
    img = colorear(m_disco(d), BLANCO)
    img.alpha_composite(colorear(mi("reloj", d, minutos), VERDE))
    return Sp(img)


@functools.lru_cache(None)
def sp_cartel():
    S = 420
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    m, d = lienzo(S, S)
    k = SS
    d.line([(S / 2 * k, S / 2 * k), (90 * k, (S / 2 + 64) * k)], fill=255, width=6 * k)
    d.line([(S / 2 * k, S / 2 * k), ((S - 90) * k, (S / 2 + 64) * k)], fill=255, width=6 * k)
    img.alpha_composite(colorear(red(m), TEXTO))
    cart = con_sombra(m_rrect(300, 104, 22), ROJO, radio=10, dy=6, opac=0.2)
    img.alpha_composite(cart.img, (S // 2 - 150 - cart.ox, S // 2 + 60 - cart.oy))
    t = sp_texto("CERRADO", 48, 800, BLANCO)
    img.alpha_composite(t.img, (int(S / 2 - t.w / 2 - t.ox), int(S / 2 + 60 + 52 - t.vc - t.oy)))
    img.alpha_composite(sp_circulo(20, GRIS).img, (S // 2 - 10, S // 2 - 10))
    return Sp(img)


def escena2(t, ctx):
    c = fondo(t)
    T = ctx["T"]
    inicios = [T["A"], T["B"], T["C"]]
    for k, (x0, y0, w, h) in enumerate(PANELES):
        a = aparecer(t, inicios[k] - 0.12, 0.35)
        if a <= 0:
            continue
        y0 = y0 + (1 - eo(a)) * 70 + flotar(t, 4, 3.8, k * 1.7)
        pegar(c, sp_tarjeta(w, h, 48), x0, y0, a=a)
        if k == 0:
            ap = 10 + 26 * (0.5 + 0.5 * math.sin(2 * math.pi * 2.6 * (t - T["A"])))
            pegar(c, sp_icono("tijeras", 300, VERDE, int(ap)), x0 + 120, y0 + 50, a=a)
            sonando = ((t - T["A"]) % 1.1) < 0.6 and t > T["A"] + 0.2
            vx = 6 * math.sin(2 * math.pi * 22 * t) if sonando else 0
            mov = sp_movil_chico(170, 290)
            pegar(c, mov, x0 + 590 + vx, y0 + 70, a=a)
            if sonando:
                pegar(c, sp_icono("ondas", 330, VERDE, 0), x0 + 675 - 165 + vx, y0 + 215 - 165, a=a * 0.85)
            n = 1 + (t > T["A"] + 0.5) + (t > T["A"] + 1.0)
            pegar(c, sp_insignia(n), x0 + 720, y0 + 40, 0.62 * pop(t, T["A"] + 0.15), a)
            texto(c, "Mientras atiendes…", x0 + w / 2, y0 + 425, 60, 700, TEXTO, al="c", va="m", a=a)
        elif k == 1:
            mins = 19 * 60 + 220 * eio((t - T["B"]) / 1.6) + max(0, t - T["B"] - 1.6) * 6
            pegar(c, sp_reloj_grande(280, int(mins)), x0 + 90, y0 + 60, a=a)
            pegar(c, sp_icono("luna", 120, VERDE, 0), x0 + 470, y0 + 50, a=a)
            for i, (sx, sy, sz) in enumerate([(610, 60, 46), (430, 190, 34), (600, 200, 30)]):
                tw = 0.55 + 0.45 * math.sin(2 * math.pi * t * 0.9 + i * 2)
                pegar(c, sp_icono("estrella", sz, VERDE_MEDIO, 0), x0 + sx, y0 + sy, a=a * tw)
            ang = 7 * math.sin(2 * math.pi * 0.8 * (t - T["B"])) * math.exp(-max(0, t - T["B"]) * 0.5)
            cart = sp_cartel()
            img = cart.img.rotate(ang, resample=RS.BICUBIC)
            pegar(c, Sp(img), x0 + 705 - 210, y0 + 95 - 210, a=a)
            texto(c, "…o cuando ya cerraste", x0 + w / 2, y0 + 425, 60, 700, TEXTO, al="c", va="m", a=a)
        else:
            b1 = sp_burbuja("¿Hola? ¿Hay espacio hoy?", "izq", VERDE_CLARO, 600)
            b2 = sp_burbuja("¿Atienden?", "izq", VERDE_CLARO, 600)
            pegar(c, b1, x0 + 50, y0 + 50, pop(t, T["C"]), a * cl((t - T["C"]) / 0.12), ancla=(0, 0))
            pegar(c, b2, x0 + 50, y0 + 170, pop(t, T["C"] + 0.55), a * cl((t - T["C"] - 0.55) / 0.12), ancla=(0, 0))
            a3 = aparecer(t, T["D"], 0.3)
            pegar(c, sp_icono("reloj", 44, GRIS, 130), x0 + 54, y0 + 296, a=a3)
            texto(c, "Sin respuesta…", x0 + 112, y0 + 318, 40, 500, GRIS, va="m", a=a3)
            sale = eio((t - T["E"]) / 0.9)
            ap = pop(t, T["C"] + 0.25) * (1 - 0.1 * sale)
            pegar(c, sp_avatar(210, "persona", VERDE_CLARO, VERDE), x0 + 640 + sale * 120, y0 + 60, ap, a * (1 - 0.65 * sale))
            af = aparecer(t, T["E"], 0.3)
            pegar(c, sp_icono("flecha", 110, ROJO, 0), x0 + 690 + 40 * eo(af), y0 + 290, a=af * (1 - 0.6 * sale))
            texto(c, "¡Y se van a otro lado!", x0 + w / 2, y0 + 540 + (1 - eo(af)) * 20, 62, 800, ROJO, al="c", va="m", a=af)
    return c


# ----------------------------------------------------------------- escena 3: la solucion
def escena3(t, ctx):
    c = fondo(t, oscuro=True)
    T = ctx["T"]
    s = pop(t, T["logo"], 0.4)
    pegar(c, sp_logo(250, CREMA, VERDE, 0.25), W / 2 - 125, 420 + flotar(t, 8, 3.2), s)
    a = aparecer(t, T["nombre"], 0.4)
    size = ajustar("Novara Software", 104, 800, 960)
    texto(c, "Novara Software", W / 2, 772 + (1 - eo(a)) * 40, size, 800, CREMA, al="c", va="m", a=a)
    a = aparecer(t, T["frase"], 0.4)
    size = ajustar("Tu WhatsApp responde solo", 66, 600, 960)
    texto(c, "Tu WhatsApp responde solo", W / 2, 897 + (1 - eo(a)) * 30, size, 600, VERDE_CLARO, al="c", va="m", a=a)
    for i, (txt, ico, tk) in enumerate([("Responde", "chat", "responde"), ("Agenda", "calendario", "agenda"),
                                        ("24 horas", "reloj", "horas")]):
        sp = sp_pastilla(txt, 58, 700, CREMA, VERDE, icono=ico, h=120, sombra=0.2)
        yy = 1080 + i * 160 + flotar(t, 5, 3.0, i)
        pegar(c, sp, W / 2 - sp.w / 2, yy, pop(t, T[tk], 0.34), cl((t - T[tk]) / 0.12))
    return c


# ----------------------------------------------------------------- escenas 4 y 5: chat + agenda
CH = (70, 320, 940, 930)
AG = (70, 1282, 940, 580)
HORAS = ["15:00", "16:00", "17:00", "18:00"]
SLOT_X, SLOT_W, SLOT_H, FILA_Y, FILA_P = 210, 690, 74, 112, 88

MSG_P1 = [
    dict(k="m1", lado="der", texto="Hola, ¿hay espacio mañana a las 4?"),
    dict(k="m2", lado="izq", texto="¡Hola! Mañana a las 16:00 está libre. ¿A nombre de quién la agendo?", tip=True),
    dict(k="m3", lado="der", texto="Carlos"),
    dict(k="m4", lado="izq", texto="Listo Carlos ✅ Tu cita quedó para mañana a las 16:00.", tip=True),
]
MSG_P2 = [
    dict(k="m5", lado="der", texto="Mejor cámbiala a las 5"),
    dict(k="m6", lado="izq", texto="Hecho ✅ Ahora es a las 17:00. La de las 16:00 ya quedó libre.", tip=True),
]


@functools.lru_cache(None)
def sp_agenda_base(titulo):
    x, y, w, h = AG
    sp = sp_tarjeta(w, h, 44)
    img = sp.img.copy()
    img.alpha_composite(colorear(mi("calendario", 50), VERDE), (sp.ox + 40, sp.oy + 34))
    t = sp_texto(titulo, 44, 700, TEXTO)
    img.alpha_composite(t.img, (sp.ox + 106 - t.ox, int(sp.oy + 59 - t.vc - t.oy)))
    for i, hora in enumerate(HORAS):
        t = sp_texto(hora, 42, 700, TEXTO)
        img.alpha_composite(t.img, (sp.ox + 40 - t.ox, int(sp.oy + FILA_Y + i * FILA_P + SLOT_H / 2 - t.vc - t.oy)))
    return Sp(img, sp.ox, sp.oy, w, h)


@functools.lru_cache(None)
def sp_slot(tipo, txt="", resaltado=False):
    w, h = SLOT_W, SLOT_H
    if tipo == "ocupado":
        img = colorear(m_rrect(w, h, 22), BEIGE)
        izq, der, ci, cd = txt, "Ocupado", TEXTO, GRIS
    elif tipo == "libre":
        img = colorear(m_rrect(w, h, 22), (0xF4, 0xFA, 0xF6) if resaltado else BLANCO)
        img.alpha_composite(colorear(m_borde(w, h, 22, 5 if resaltado else 3), VERDE if resaltado else VERDE_CLARO))
        izq, der, ci, cd = "Libre", "", VERDE, VERDE
    else:
        img = colorear(m_rrect(w, h, 22), VERDE)
        izq, der, ci, cd = txt, "WhatsApp", BLANCO, VERDE_CLARO
    t = sp_texto(izq, 40, 700, ci)
    img.alpha_composite(t.img, (28 - t.ox, int(h / 2 - t.vc - t.oy)))
    if der:
        t = sp_texto(der, 40, 500, cd)
        xr = w - 28 - t.w
        img.alpha_composite(t.img, (int(xr - t.ox), int(h / 2 - t.vc - t.oy)))
        if tipo == "cita":
            img.alpha_composite(colorear(mi("chat", 44), VERDE_CLARO), (int(xr - 56), int(h / 2 - 22)))
    if resaltado:
        img.alpha_composite(sp_emoji("✅", 46).img, (w - 28 - 46, (h - 46) // 2))
    return Sp(img)


def slot_xy(i, dy):
    return AG[0] + SLOT_X, AG[1] + dy + FILA_Y + i * FILA_P


def pies(c, t, dy, grupos):
    """Pastillas de mensaje clave al pie de la agenda.
    grupos = [[(texto, t_in, t_out), ...], ...]; cada grupo se centra con todas sus pastillas
    (aunque aun no aparezcan) para que nada salte de lugar."""
    for grupo in grupos:
        sps = [sp_pastilla(tx, 40, 700, VERDE_CLARO, VERDE, icono="check_circulo", h=72, sombra=0.0)
               for tx, _, _ in grupo]
        x = W / 2 - (sum(s.w for s in sps) + 20 * (len(sps) - 1)) / 2
        for sp, (tx, ti, to) in zip(sps, grupo):
            a = cl((t - ti) / 0.12) * (1 - cl((t - to) / 0.3))
            if t >= ti and a > 0:
                pegar(c, sp, x, AG[1] + dy + 484, pop(t, ti, 0.3), a)
            x += sp.w + 20


def escena_chat_agenda(t, ctx, paso):
    tf = t + ctx.get("t_continuo", 0.0)   # reloj continuo entre escena 4 y 5
    c = fondo(tf)
    T = ctx["T"]
    if paso == 1:
        encabezado(c, t, "PASO 1", "El cliente pide su cita")
        msgs = [dict(m, t=T[m["k"]]) for m in MSG_P1]
    else:
        encabezado(c, t, "PASO 2", "¿Quiere otra hora?")
        msgs = [dict(m, t=-99.0) for m in MSG_P1] + [dict(m, t=T[m["k"]]) for m in MSG_P2]
    dy1 = flotar(tf, 4, 3.6, 0.0)
    dy2 = flotar(tf, 4, 3.6, 1.2)
    Chat(*CH, "Tu Barbería", "tijeras", msgs).dibujar(c, t, dy1)
    pegar(c, sp_agenda_base("Tu agenda · mañana"), AG[0], AG[1] + dy2)
    ocup = {0: "Ana · Corte", 3: "Rosa · Tinte"}
    for i, txt in ocup.items():
        pegar(c, sp_slot("ocupado", txt), *slot_xy(i, dy2))
    verde = sp_slot("cita", "Carlos")
    if paso == 1:
        # escaneo de la agenda: la lupa baja de 15:00 a 16:00
        for i in (1, 2):
            res = (i == 1 and T["scan_fin"] <= t < T["fill"]) or (T["libres"] <= t < T["fill"] + (1.0 if i == 2 else 0))
            pegar(c, sp_slot("libre", "", res), *slot_xy(i, dy2))
        if T["scan"] - 0.1 <= t < T["scan_fin"] + 0.35:
            p = eio((t - T["scan"]) / (T["scan_fin"] - T["scan"]))
            a = aparecer(t, T["scan"] - 0.1, 0.15) * (1 - aparecer(t, T["scan_fin"] + 0.1, 0.25))
            sx, sy = slot_xy(0, dy2)
            yy = sy + p * FILA_P
            pegar(c, Sp(colorear(m_rrect(SLOT_W, SLOT_H, 22).point(lambda v: int(v * 0.35)), VERDE_CLARO)), sx, yy, a=a)
            pegar(c, sp_icono("lupa", 64, VERDE, 0), sx + SLOT_W - 140, yy + 6, a=a)
        if t >= T["fill"]:
            sx, sy = slot_xy(1, dy2)
            pegar(c, verde, sx, sy, pop(t, T["fill"], 0.3), cl((t - T["fill"]) / 0.1))
        pies(c, t, dy2, [[("Solo horarios libres", T["libres"], 99)]])
    else:
        mv = (t - T["mover"]) / (T["mover_fin"] - T["mover"])
        sx1, sy1 = slot_xy(1, dy2)
        sx2, sy2 = slot_xy(2, dy2)
        destello = t >= T["mover"] and t < T["mover_fin"] + 0.8
        pegar(c, sp_slot("libre", "", destello and t >= T["mover"] + 0.15), sx1, sy1)
        if t < T["mover_fin"]:
            pegar(c, sp_slot("libre"), sx2, sy2)
        if t < T["mover"]:
            pegar(c, verde, sx1, sy1)
        elif t < T["mover_fin"]:
            p = eio(mv)
            lev = math.sin(math.pi * cl(mv)) * 0.05
            sp = con_sombra(m_rrect(SLOT_W, SLOT_H, 22), VERDE, radio=14, dy=10, opac=0.3 * math.sin(math.pi * cl(mv)))
            img = sp.img.copy()
            img.alpha_composite(verde.img, (sp.ox, sp.oy))
            pegar(c, Sp(img, sp.ox, sp.oy, SLOT_W, SLOT_H), sx1, lerp(sy1, sy2, p), 1 + lev)
        else:
            pegar(c, verde, sx2, sy2, 1 + 0.06 * math.sin(math.pi * cl((t - T["mover_fin"]) / 0.25)))
        pies(c, t, dy2, [[("Solo horarios libres", -99, T["cruces"] - 0.35)],
                         [("Sin cruces", T["cruces"], 99), ("Sin doble reserva", T["doble"], 99)]])
    return c


def escena4(t, ctx):
    return escena_chat_agenda(t, ctx, 1)


def escena5(t, ctx):
    return escena_chat_agenda(t, ctx, 2)


# ----------------------------------------------------------------- escena 6: tu panel
TEL6 = (130, 330, 820, 1510)
AGENDA_HOY = [("09:00", "Jhonny Rojas", "Corte", "WhatsApp"), ("10:30", "María López", "Tinte", "Manual"),
              ("12:00", "Daniela Suárez", "Uñas", "WhatsApp"), ("15:00", "Ana Vargas", "Corte", "Manual"),
              ("17:00", "Luis Pinto", "Barba", "WhatsApp"), ("18:30", "Rosa Méndez", "Peinado", "WhatsApp")]
CHATS = [("Carlos", "Mejor cámbiala a las 5", True), ("Jhonny Rojas", "¡Gracias, ahí estaré!", False),
         ("Daniela Suárez", "¿Hacen uñas acrílicas?", True), ("María López", "Perfecto, nos vemos", False),
         ("Luis Pinto", "¿Atienden el domingo?", True)]
FILA6_H = 160
LISTA_Y = 420


@functools.lru_cache(None)
def sp_etiqueta(tipo, realce=False):
    if tipo == "WhatsApp":
        return sp_pastilla("WhatsApp", 40, 600, VERDE_CLARO, VERDE, icono="chat", h=62, sombra=0.0, padx=20)
    return sp_pastilla("Manual", 40, 600, VERDE if realce else (0xEF, 0xE8, 0xD6), BLANCO if realce else TEXTO,
                       icono="lapiz", h=62, sombra=0.0, padx=20)


@functools.lru_cache(None)
def sp_fila_agenda(i, realce=False):
    hora, nombre, serv, tag = AGENDA_HOY[i]
    sw = TEL6[2] - 2 * BISEL
    img = Image.new("RGBA", (sw, FILA6_H), (0xFF, 0xFF, 0xFF, 0))
    if realce:
        img.alpha_composite(colorear(m_rrect(sw - 24, FILA6_H - 16, 26), (0xFB, 0xF3, 0xDC)), (12, 8))
    for txt, x, y, sz, pz, col in [(hora, 30, 30, 42, 700, TEXTO), (nombre, 204, 26, 42, 700, TEXTO),
                                   (serv, 204, 86, 40, 400, GRIS)]:
        t = sp_texto(txt, sz, pz, col)
        img.alpha_composite(t.img, (x - t.ox, y - t.oy))
    barra = colorear(m_rrect(8, 100, 4), VERDE if tag == "WhatsApp" else (0xC8, 0xA9, 0x5E))
    img.alpha_composite(barra, (178, 30))
    et = sp_etiqueta(tag, realce)
    img.alpha_composite(et.img, (sw - 30 - et.w - et.ox, 78 - et.oy))
    if i < len(AGENDA_HOY) - 1:
        img.alpha_composite(colorear(Image.new("L", (sw - 60, 2), 255), LINEA), (30, FILA6_H - 2))
    return Sp(img)


@functools.lru_cache(None)
def sp_fila_chat(i):
    nombre, prev, nuevo = CHATS[i]
    sw = TEL6[2] - 2 * BISEL
    img = Image.new("RGBA", (sw, FILA6_H), (0xFF, 0xFF, 0xFF, 0))
    img.alpha_composite(sp_avatar_iniciales(100, nombre).img, (30, 30))
    t = sp_texto(nombre, 42, 700, TEXTO)
    img.alpha_composite(t.img, (156 - t.ox, 28 - t.oy))
    prev = recortar_txt(prev, 40, 400, sw - 156 - 90)
    t = sp_texto(prev, 40, 400, GRIS)
    img.alpha_composite(t.img, (156 - t.ox, 88 - t.oy))
    if nuevo:
        img.alpha_composite(sp_circulo(30, VERDE).img, (sw - 62, 102))
    if i < len(CHATS) - 1:
        img.alpha_composite(colorear(Image.new("L", (sw - 60, 2), 255), LINEA), (30, FILA6_H - 2))
    return Sp(img)


@functools.lru_cache(None)
def pantalla6_base():
    sw, sh = TEL6[2] - 2 * BISEL, TEL6[3] - 2 * BISEL
    p = Image.new("RGB", (sw, sh), (0xFA, 0xF8, 0xF3))
    p.paste(VERDE, (0, 0, sw, 300))
    texto(p, "Agenda de hoy", 46, 110, 56, 800, BLANCO)
    texto(p, "Viernes 20 de noviembre", 46, 196, 40, 500, VERDE_CLARO)
    sp = sp_pastilla("6 citas", 40, 700, BLANCO, VERDE, h=66, sombra=0.0, padx=24)
    pegar(p, sp, sw - 40 - sp.w, 112)
    p.paste(BLANCO, (0, 300, sw, LISTA_Y))
    p.paste(LINEA, (0, LISTA_Y - 2, sw, LISTA_Y))
    return p


def escena6(t, ctx):
    c = fondo(t)
    T = ctx["T"]
    encabezado(c, t, "PASO 3", "Tu agenda en un panel")
    x, y, w, h = TEL6
    y += flotar(t, 5, 3.8)
    sw, sh = w - 2 * BISEL, h - 2 * BISEL
    pant = pantalla6_base().copy()
    cambio = eio((t - T["chats"] - 0.12) / 0.45)
    # pestanas
    tab_x = [sw * 0.25, sw * 0.75]
    for i, nom in enumerate(["Agenda", "Chats"]):
        activo = (cambio < 0.5) == (i == 0)
        texto(pant, nom, tab_x[i], 360, 44, 700, VERDE if activo else GRIS, al="c", va="m")
    pegar(pant, sp_insignia(3), tab_x[1] + 30, 340 - 66 - 18, 0.38)
    ux = lerp(tab_x[0], tab_x[1], cambio)
    pegar(pant, Sp(colorear(m_rrect(200, 8, 4), VERDE)), ux - 100, LISTA_Y - 10)
    # lista: agenda y chats deslizandose
    lista = Image.new("RGB", (sw, sh - LISTA_Y), (0xFA, 0xF8, 0xF3))
    off = -sw * cambio
    for i in range(len(AGENDA_HOY)):
        ti = T["filas"] + i * 0.11
        a = aparecer(t, ti, 0.3)
        realce = AGENDA_HOY[i][3] == "Manual" and t >= T["manual"]
        sp = sp_fila_agenda(i, realce)
        pul = 1.0
        if realce:
            pul = 1 + 0.025 * math.sin(math.pi * cl((t - T["manual"]) / 0.35))
        pegar(lista, sp, off + (1 - eo(a)) * 80, 10 + i * FILA6_H, pul, a)
        if realce and t >= T["respeta"]:
            pegar(lista, sp_emoji("✅", 48), off + sw - 30 - sp_etiqueta("Manual", True).w - 64,
                  10 + i * FILA6_H + 85, pop(t, T["respeta"] + i * 0.05))
    if cambio > 0:
        for i in range(len(CHATS)):
            pegar(lista, sp_fila_chat(i), off + sw + 40 * (1 - cambio) * (i + 1) / 3, 10 + i * FILA6_H)
    pant.paste(lista, (0, LISTA_Y))
    # toque en la pestana "Chats"
    lt = t - T["chats"]
    if 0 <= lt < 0.6:
        r = 40 + 120 * eo(lt / 0.6)
        pegar(pant, sp_circulo(int(2 * r), VERDE), tab_x[1] - r, 360 - r, a=0.3 * (1 - lt / 0.6))
        pegar(pant, sp_circulo(70, VERDE), tab_x[1] - 35, 325, a=0.45 * (1 - cl(lt / 0.4)))
    pegar(pant, Sp(colorear(m_rrect(220, 10, 5), (0x30, 0x38, 0x33))), sw / 2 - 110, sh - 30)
    dibujar_telefono(c, x, y, w, h, pant)
    return c


# ----------------------------------------------------------------- escena 7: feriados
CH7 = (70, 320, 940, 880)
CAL = (70, 1250, 940, 590)
DIAS = [("Hoy", "20"), ("Mañana", "21"), ("Dom", "22"), ("Lun", "23")]
MSG_P4 = [dict(k="m1", lado="der", texto="Mañana es feriado, no atendemos"),
          dict(k="m2", lado="izq", texto="Listo ⛔ Cerré mañana. Si alguien pide cita, le aviso que es feriado.", tip=True)]


@functools.lru_cache(None)
def sp_dia(i, cerrado=False):
    nom, num = DIAS[i]
    w, h = 200, 270
    img = colorear(m_rrect(w, h, 30), ROJO_CLARO if cerrado else BEIGE)
    if i == 1 and not cerrado:
        img.alpha_composite(colorear(m_borde(w, h, 30, 4), VERDE))
    for txt, y, sz, pz, col in [(nom, 52, 40, 700, TEXTO if i == 1 else GRIS), (num, 132, 80, 800, TEXTO),
                                ("Feriado" if cerrado else "Abierto", 222, 40, 700, ROJO if cerrado else VERDE)]:
        t = sp_texto(txt, sz, pz, col)
        img.alpha_composite(t.img, (int(w / 2 - t.w / 2 - t.ox), int(y - t.vc - t.oy)))
    return Sp(img)


def escena7(t, ctx):
    c = fondo(t)
    T = ctx["T"]
    encabezado(c, t, "PASO 4", "¿Feriado o viaje?")
    dy1 = flotar(t, 4, 3.6, 0.0)
    dy2 = flotar(t, 4, 3.6, 1.2)
    msgs = [dict(m, t=T[m["k"]]) for m in MSG_P4]
    Chat(*CH7, "Tu asistente", "chat", msgs).dibujar(c, t, dy1)
    x, y, w, h = CAL
    y += dy2
    pegar(c, sp_agenda_base_cal(), x, y)
    for i in range(4):
        dx = x + 40 + i * 220
        if i == 1 and t >= T["sello"]:
            pegar(c, sp_dia(i, True), dx, y + 118)
            s = lerp(2.2, 1.0, eo((t - T["sello"]) / 0.22))
            pegar(c, sp_emoji("⛔", 60), dx + 152, y + 118 - 36, s, cl((t - T["sello"]) / 0.1))
        else:
            pegar(c, sp_dia(i), dx, y + 118)
    sp = sp_pastilla("Ese día no se agenda", 40, 700, ROJO_CLARO, ROJO, icono="prohibido", h=72, sombra=0.0)
    pegar(c, sp, W / 2 - sp.w / 2, y + 438, pop(t, T["nota"], 0.3), cl((t - T["nota"]) / 0.12))
    return c


@functools.lru_cache(None)
def sp_agenda_base_cal():
    x, y, w, h = CAL
    sp = sp_tarjeta(w, h, 44)
    img = sp.img.copy()
    img.alpha_composite(colorear(mi("calendario", 50), VERDE), (sp.ox + 40, sp.oy + 34))
    t = sp_texto("Tu semana", 44, 700, TEXTO)
    img.alpha_composite(t.img, (sp.ox + 106 - t.ox, int(sp.oy + 59 - t.vc - t.oy)))
    return Sp(img, sp.ox, sp.oy, w, h)


# ----------------------------------------------------------------- escena 8: cierre
def escena8(t, ctx):
    c = fondo(t)
    T = ctx["T"]
    pegar(c, sp_logo(260, VERDE, CREMA), W / 2 - 130, 470 + flotar(t, 8, 3.2), pop(t, T["logo"], 0.4))
    a = aparecer(t, T["nombre"], 0.4)
    size = ajustar("Novara Software", 108, 800, 960)
    texto(c, "Novara Software", W / 2, 850 + (1 - eo(a)) * 40, size, 800, VERDE, al="c", va="m", a=a)
    a = aparecer(t, T["frase"], 0.4)
    size = ajustar("Tu negocio, siempre atendiendo", 64, 500, 960)
    texto(c, "Tu negocio, siempre atendiendo", W / 2, 960 + (1 - eo(a)) * 30, size, 500, TEXTO, al="c", va="m", a=a)
    sp = sp_pastilla("Demo gratis", 66, 800, VERDE, BLANCO, h=146, sombra=0.25, padx=80)
    lat = 1 + 0.03 * max(0.0, math.sin(2 * math.pi * (t - T["cta"]) / 1.3)) if t > T["cta"] + 0.5 else 1
    pegar(c, sp, W / 2 - sp.w / 2, 1180, pop(t, T["cta"], 0.36) * lat, cl((t - T["cta"]) / 0.12))
    a = aparecer(t, T["url"], 0.4)
    size = ajustar("novarasoftware.up.railway.app", 58, 700, 960)
    texto(c, "novarasoftware.up.railway.app", W / 2, 1420 + (1 - eo(a)) * 24, size, 700, VERDE, al="c", va="m", a=a)
    return c


DIBUJAR = {1: escena1, 2: escena2, 3: escena3, 4: escena4, 5: escena5, 6: escena6, 7: escena7, 8: escena8}


# ----------------------------------------------------------------- plan de tiempos
def tiempos_internos(i, F, Ff, dur, voz):
    if i == 1:
        # momento en que se dice cada palabra del titulo (proporcional a los caracteres)
        palabras = ["¿Cuántos", "clientes", "pierdes", "por", "no", "contestar", "a", "tiempo?"]
        idx, pos = [], 0
        for p in palabras:
            m = re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", voz[pos:])
            idx.append(pos + m.start())
            pos += m.end()
        tw = [F[0] + (Ff[0] - F[0]) * k / len(voz) for k in idx]
        lleg = [0.35 + 4.15 * (k / 13) ** 0.72 for k in range(14)]
        return dict(palabras=tw, llegadas=lleg)
    if i == 2:
        return dict(A=F[0], B=F[1], C=F[2], D=F[3], E=F[4])
    if i == 3:
        d = Ff[1] - F[1]
        return dict(logo=0.32, nombre=0.5, frase=F[1], responde=F[1] + d * 11 / 34, agenda=F[1] + d * 22 / 34, horas=F[2])
    if i == 4:
        return dict(m1=0.45, scan=1.9, scan_fin=2.55, m2=3.1, libres=4.25, m3=4.55, m4=5.4, fill=5.55)
    if i == 5:
        return dict(m5=0.45, m6=1.45, mover=2.2, mover_fin=3.05, cruces=F[2], doble=F[3])
    if i == 6:
        return dict(filas=0.45, manual=F[1] + 0.1, respeta=F[1] + 1.75, chats=F[2] + 0.1)
    if i == 7:
        return dict(m1=0.85, m2=2.25, sello=3.3, nota=3.85)
    if i == 8:
        return dict(logo=0.3, nombre=0.45, frase=0.95, cta=F[1], url=F[1] + 0.35)


def eventos_escena(i, T):
    ev = []
    if i == 1:
        ev += [(ta, "notificacion") for ta in T["llegadas"]]
    elif i == 2:
        ev += [(T["A"] + 0.2, "vibra"), (T["A"] + 1.3, "vibra"), (T["C"], "pop"), (T["C"] + 0.55, "pop"),
               (T["E"], "sale")]
    elif i == 3:
        ev += [(T["logo"], "campana"), (T["responde"], "pop"), (T["agenda"], "pop"), (T["horas"], "pop")]
    elif i == 4:
        ev += [(T["m1"], "pop"), (T["m2"], "pop"), (T["m3"], "pop"), (T["m4"], "pop"), (T["fill"], "exito"),
               (T["libres"], "pop_suave")]
    elif i == 5:
        ev += [(T["m5"], "pop"), (T["m6"], "pop"), (T["mover"], "desliza"), (T["mover_fin"], "exito"),
               (T["cruces"], "pop_suave"), (T["doble"], "pop_suave")]
    elif i == 6:
        ev += [(T["filas"] + k * 0.11, "tic") for k in range(6)]
        ev += [(T["manual"], "pop_suave"), (T["chats"], "toque")]
    elif i == 7:
        ev += [(T["m1"], "pop"), (T["m2"], "pop"), (T["sello"], "sello"), (T["nota"], "pop_suave")]
    elif i == 8:
        ev += [(T["logo"], "campana"), (T["cta"], "pop")]
    return ev


def calcular_plan():
    with open(os.path.join(BUILD, "voz", "tiempos.json")) as f:
        tiempos = json.load(f)
    escenas, cuadros, eventos = [], 0, []
    for e in ESCENAS:
        v = tiempos[str(e["id"])]
        dur = max(e["minimo"], VOZ_RETARDO + v["duracion"] + 0.5)
        n = int(round(dur * FPS))
        F = [VOZ_RETARDO + f["inicio"] for f in v["frases"]]
        Ff = [VOZ_RETARDO + f["fin"] for f in v["frases"]]
        s = dict(id=e["id"], nombre=e["nombre"], inicio=cuadros / FPS, frames=n, dur=n / FPS,
                 voz_inicio=cuadros / FPS + VOZ_RETARDO, F=F, Ff=Ff, transicion=e["transicion"])
        s["T"] = tiempos_internos(e["id"], F, Ff, n / FPS, e["tts"])
        if e["id"] == 5:   # la escena 5 es continuacion directa de la 4
            s["t_continuo"] = escenas[-1]["dur"]
            s["z_base"] = 1.0
        if e["id"] > 1:
            eventos.append((round(s["inicio"] - 0.06, 3), "whoosh"))
        eventos += [(round(s["inicio"] + tt, 3), tipo) for tt, tipo in eventos_escena(e["id"], s["T"])]
        escenas.append(s)
        cuadros += n
    plan = dict(fps=FPS, frames=cuadros, total=cuadros / FPS, escenas=escenas, eventos=sorted(eventos))
    with open(os.path.join(BUILD, "plan.json"), "w") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)
    for s in escenas:
        print(f"escena {s['id']} {s['nombre']:9} inicio {s['inicio']:6.2f}s  dur {s['dur']:5.2f}s  ({s['frames']} cuadros)")
    print(f"TOTAL {plan['total']:.2f} s, {cuadros} cuadros, {len(plan['eventos'])} efectos de sonido")
    return plan


def cargar_plan():
    with open(os.path.join(BUILD, "plan.json")) as f:
        return json.load(f)


# ----------------------------------------------------------------- render
def cuadro_solo(plan, i, t):
    s = plan["escenas"][i - 1]
    img = DIBUJAR[i](t, s)
    z = 1 + 0.022 * (s.get("z_base", 0.0) + cl(t / s["dur"], 0, 1.1))   # zoom lento constante
    if z > 1.0005:
        img = img.transform((W, H), Image.Transform.AFFINE,
                            (1 / z, 0, W / 2 * (1 - 1 / z), 0, 1 / z, H / 2 * (1 - 1 / z)), resample=RS.BICUBIC)
    return img


def cuadro(plan, i, t):
    img = cuadro_solo(plan, i, t)
    if t < TRANSICION:
        e = eio(t / TRANSICION)
        if i == 1:
            return Image.blend(fondo(0), img, e)
        prev = cuadro_solo(plan, i - 1, plan["escenas"][i - 2]["dur"] + t)
        if plan["escenas"][i - 1]["transicion"] == "fundido":
            return Image.blend(prev, img, e)
        comp = Image.new("RGB", (W, H))
        comp.paste(prev, (round(-W * e), 0))
        comp.paste(img, (round(W * (1 - e)), 0))
        return comp
    return img


def render_escena(i):
    plan = cargar_plan()
    s = plan["escenas"][i - 1]
    os.makedirs(os.path.join(BUILD, "escenas"), exist_ok=True)
    out = os.path.join(BUILD, "escenas", f"escena_{i}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-level", "4.1", "-g", str(FPS * 2), "-bf", "2", "-r", str(FPS), out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(s["frames"]):
        p.stdin.write(cuadro(plan, i, f / FPS).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg fallo en escena {i}")
    return i, s["frames"]


FOTOS = {1: [4.9], 2: [7.6], 3: [4.9], 4: [6.9, 2.3], 5: [7.2, 2.6], 6: [5.0, 8.0], 7: [5.9], 8: [6.9]}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "todo"
    if cmd == "plan":
        calcular_plan()
    elif cmd == "fotos":
        plan = cargar_plan()
        d = os.path.join(BUILD, "fotos")
        os.makedirs(d, exist_ok=True)
        ids = [int(a) for a in sys.argv[2:]] or list(FOTOS)
        for i in ids:
            for t in FOTOS[i]:
                t = min(t, plan["escenas"][i - 1]["dur"] - 1 / FPS)
                ruta = os.path.join(d, f"escena_{i}_{t:.1f}s.png")
                cuadro(plan, i, t).save(ruta)
                print(ruta)
    elif cmd == "render":
        ids = [int(a) for a in sys.argv[2:]] or list(DIBUJAR)
        ids.sort(key=lambda k: -cargar_plan()["escenas"][k - 1]["frames"])
        with Pool(min(4, os.cpu_count() or 2)) as pool:
            for i, n in pool.imap_unordered(render_escena, ids):
                print(f"escena {i} lista ({n} cuadros)", flush=True)


if __name__ == "__main__":
    main()

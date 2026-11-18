"""kolam.synth -- make realistic test images from generated kolams (rotation, noise, blur,
uneven lighting, hand-drawn-style warping, inverted colours)."""
from __future__ import annotations

import cv2
import numpy as np


def elastic_warp(img, amplitude=1.5, sigma=18, seed=0, bg=(0, 0, 0)):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dx *= amplitude / (dx.std() + 1e-9); dy *= amplitude / (dy.std() + 1e-9)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderValue=bg)


def perturb(img, bg, angle=0.0, noise=0.0, blur=0.0, shade=0.0, warp=0.0, seed=0):
    out = img.copy()
    h, w = out.shape[:2]
    if warp:
        out = elastic_warp(out, warp, seed=seed, bg=bg)
    if angle:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderValue=bg)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if shade:   # smooth illumination gradient
        gx = np.linspace(1 - shade, 1 + shade * 0.3, w)[None, :, None]
        out = np.clip(out.astype(np.float32) * gx, 0, 255).astype(np.uint8)
    if noise:
        out = np.clip(out.astype(np.float32) + np.random.default_rng(seed).normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    return out

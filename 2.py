# -*- coding: utf-8 -*-
"""
water_physics_raytrace.py
物理折射版（Snell + Ray Tracing）

特点：
- 不用MLE拟合畸变
- 不修改K
- 显式建模：
    空气 → 玻璃 → 水
"""

import numpy as np
import cv2


# =========================
# 折射率（必须）
# =========================
N_AIR = 1.0
N_GLASS = 1.5
N_WATER = 1.333


# =========================
# 像素 → 相机射线
# =========================
def pixel_to_ray(K, u, v):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = np.array([x, y, 1.0])
    return ray / np.linalg.norm(ray)


# =========================
# Snell折射
# =========================
def refract(I, N, n1, n2):
    I = I / np.linalg.norm(I)
    N = N / np.linalg.norm(N)

    cosi = -np.dot(N, I)
    eta = n1 / n2

    k = 1 - eta**2 * (1 - cosi**2)
    if k < 0:
        return None

    T = eta * I + (eta * cosi - np.sqrt(k)) * N
    return T / np.linalg.norm(T)


# =========================
# 三段折射
# =========================
def ray_trace(K, u, v, glass_thickness, d0):
    ray = pixel_to_ray(K, u, v)

    # -------- 空气 → 玻璃 --------
    N1 = np.array([0,0,1])
    r1 = refract(ray, N1, N_AIR, N_GLASS)

    # 玻璃内部传播
    p1 = r1 * glass_thickness

    # -------- 玻璃 → 水 --------
    N2 = np.array([0,0,1])
    r2 = refract(r1, N2, N_GLASS, N_WATER)

    # 返回水中射线
    return r2


# =========================
# 深度求交
# =========================
def intersect_plane(ray, d):
    return d / ray[2]


# =========================
# 主函数
# =========================
def depth(K, u, v, d0, glass_t):
    ray = ray_trace(K, u, v, glass_t, d0)
    Z = intersect_plane(ray, d0)
    return Z
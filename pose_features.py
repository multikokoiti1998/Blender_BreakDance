"""poses_*.json から視点に依存しにくい姿勢特徴量（関節角度・体軸の傾き/ねじれ）を計算する共通モジュール。"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent

# MediaPipe world 座標は y が下向き。上方向ベクトル
UP = np.array([0.0, -1.0, 0.0])

FEATURES = [
    "torso_tilt",      # 体軸(腰→胸)の鉛直からの傾き [deg] 0=直立, 180=逆立ち
    "axis_twist",      # 肩ラインと腰ラインのねじれ [deg]
    "l_elbow", "r_elbow",        # 肘の曲がり (180=伸展)
    "l_shoulder", "r_shoulder",  # 腕と体幹の開き
    "l_hip", "r_hip",            # 太ももと体幹の角度 (180=伸展)
    "l_knee", "r_knee",          # 膝の曲がり (180=伸展)
    "leg_split",       # 左右の太ももの開き
]


def _angle(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def load(name):
    """JSON を読み込み、(data, world[T,J,3], image[T,J,3], detected[T]) を返す。未検出フレームは NaN。"""
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    joints = data["joints"]
    T = len(data["frames"])
    world = np.full((T, len(joints), 3), np.nan)
    image = np.full((T, len(joints), 3), np.nan)
    det = np.zeros(T, bool)
    for t, f in enumerate(data["frames"]):
        if f["detected"]:
            det[t] = True
            world[t] = [f["world"][j] for j in joints]
            image[t] = [f["image"][j] for j in joints]
    return data, world, image, det


def J(data, world, name):
    return world[:, data["joints"].index(name)]


def compute_features(data, world):
    g = lambda n: J(data, world, n)
    hip_c = (g("l_hip") + g("r_hip")) / 2
    sh_c = (g("l_shoulder") + g("r_shoulder")) / 2
    spine = sh_c - hip_c

    # ねじれ: 体軸に垂直な平面へ肩ライン/腰ラインを射影して角度を取る
    s_hat = spine / (np.linalg.norm(spine, axis=-1, keepdims=True) + 1e-9)
    def proj(v):
        return v - (v * s_hat).sum(-1, keepdims=True) * s_hat
    sh_line = proj(g("r_shoulder") - g("l_shoulder"))
    hip_line = proj(g("r_hip") - g("l_hip"))

    f = {
        "torso_tilt": _angle(spine, np.broadcast_to(UP, spine.shape)),
        "axis_twist": _angle(sh_line, hip_line),
    }
    for s in "lr":
        f[f"{s}_elbow"] = _angle(g(f"{s}_shoulder") - g(f"{s}_elbow"), g(f"{s}_wrist") - g(f"{s}_elbow"))
        f[f"{s}_shoulder"] = _angle(g(f"{s}_elbow") - g(f"{s}_shoulder"), -spine)
        f[f"{s}_hip"] = _angle(g(f"{s}_knee") - g(f"{s}_hip"), spine)
        f[f"{s}_knee"] = _angle(g(f"{s}_hip") - g(f"{s}_knee"), g(f"{s}_ankle") - g(f"{s}_knee"))
    f["leg_split"] = _angle(g("l_knee") - g("l_hip"), g("r_knee") - g("r_hip"))
    return np.stack([f[k] for k in FEATURES], axis=1)


def interp_smooth(x, win=5):
    """NaN を線形補間し、移動平均で平滑化（列ごと）。"""
    x = x.copy()
    t = np.arange(len(x))
    for c in range(x.shape[1]):
        ok = ~np.isnan(x[:, c])
        if ok.sum() >= 2:
            x[:, c] = np.interp(t, t[ok], x[ok, c])
    k = np.ones(win) / win
    pad = np.pad(x, ((win // 2, win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(x.shape[1])], axis=1)

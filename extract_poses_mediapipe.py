"""① 動画から関節座標を抽出して JSON に保存する。

    python extract_poses_mediapipe.py  # （旧版・参考用）OK.mp4 -> poses_expert.json, me.mp4 -> poses_my.json
    python extract_poses_mediapipe.py in.mp4 out.json

ブレイクダンスは逆さ・横向きの姿勢が多く、MediaPipe は「立っている人」前提なので崩れやすい。
そこで各フレームを 0/90/180/270° 回転させた4通りで推定し、体が一番まっすぐ立って見える向きの
結果を採用して元の向きに戻す（回転ごとに別の VIDEO モード推定器を使いトラッキングを保つ）。

出力 JSON の各フレーム（MediaPipe の33点すべて）:
  image : 元画像での正規化座標 x,y (0-1) + 相対深度 z
  world : 腰中心・メートル単位の3D座標 (x右 y下 z奥、元画像の向き)
  visibility, rotation(採用した回転角)
"""
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "pose_landmarker_heavy.task"


def _name(enum_name):
    n = enum_name.lower()
    return n.replace("left_", "l_").replace("right_", "r_")


JOINTS = {_name(p.name): p.value for p in vision.PoseLandmark}   # 33点
MAIN = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]

JOBS = [("OK.mp4", "poses_expert.json"), ("me.mp4", "poses_my.json")]

# 回転角 -> (cv2 の回転コード, 回転後の変位 (dx', dy') を元画像の変位 (du, dv) に戻す関数)
ROTATIONS = {
    0: (None, lambda x, y: (x, y)),
    90: (cv2.ROTATE_90_CLOCKWISE, lambda x, y: (y, -x)),
    180: (cv2.ROTATE_180, lambda x, y: (-x, -y)),
    270: (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda x, y: (-y, x)),
}


def unrotate_norm(deg, x, y):
    """回転画像での正規化座標 -> 元画像での正規化座標"""
    if deg == 0:
        return x, y
    if deg == 90:
        return y, 1 - x
    if deg == 180:
        return 1 - x, 1 - y
    return 1 - y, x


def content_box(video_path: Path):
    """縦動画を横長に書き出した時の黒帯を検出し、映像部分の (x0, y0, x1, y1) を返す"""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = None
    for i in range(0, n, max(n // 10, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
        acc = g if acc is None else np.maximum(acc, g)
    cap.release()
    cols = np.where(acc.max(0) > 20)[0]
    rows = np.where(acc.max(1) > 20)[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        # 日本語パスだと model_asset_path が開けないのでバッファで渡す
        base_options=BaseOptions(model_asset_buffer=MODEL.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.PoseLandmarker.create_from_options(options)


def score(result, rot_w, rot_h):
    """主要関節の visibility と、回転画像内で体幹がどれだけ直立しているか"""
    il = result.pose_landmarks[0]
    vis = np.mean([il[JOINTS[j]].visibility or 0.0 for j in MAIN])
    p = lambda j: np.array([il[JOINTS[j]].x * rot_w, il[JOINTS[j]].y * rot_h])
    up = (p("l_shoulder") + p("r_shoulder")) / 2 - (p("l_hip") + p("r_hip")) / 2
    ang = np.degrees(np.arccos(np.clip(-up[1] / (np.linalg.norm(up) + 1e-9), -1, 1)))  # 0=直立
    return vis - ang / 180.0


def extract(video_path: Path, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x0, y0, x1, y1 = content_box(video_path)
    width, height = x1 - x0, y1 - y0
    marks = {deg: make_landmarker() for deg in ROTATIONS}
    frames = []
    idx = 0
    prev_deg = 0
    rot_count = {deg: 0 for deg in ROTATIONS}
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        ts_ms = int(round(idx * 1000.0 / fps))
        best = None
        for deg, (code, _) in ROTATIONS.items():
            img = np.ascontiguousarray(rgb if code is None else cv2.rotate(rgb, code))
            res = marks[deg].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), ts_ms)
            if not res.pose_landmarks:
                continue
            sc = score(res, img.shape[1], img.shape[0]) + (0.05 if deg == prev_deg else 0.0)  # 向きのばたつき防止
            if best is None or sc > best[0]:
                best = (sc, deg, res)
        frame = {"frame": idx, "time": round(idx / fps, 4), "detected": best is not None}
        if best:
            _, deg, res = best
            prev_deg = deg
            rot_count[deg] += 1
            back = ROTATIONS[deg][1]
            il, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
            frame["rotation"] = deg
            frame["image"] = {}
            frame["world"] = {}
            for n, i in JOINTS.items():
                u, v = unrotate_norm(deg, il[i].x, il[i].y)
                frame["image"][n] = [round(u, 5), round(v, 5), round(il[i].z, 5)]
                wx, wy = back(wl[i].x, wl[i].y)
                frame["world"][n] = [round(wx, 5), round(wy, 5), round(wl[i].z, 5)]
            frame["visibility"] = {n: round(il[i].visibility or 0.0, 3) for n, i in JOINTS.items()}
        frames.append(frame)
        idx += 1
        if idx % 100 == 0:
            print(f"  {video_path.name}: {idx} frames", flush=True)
    cap.release()
    for m in marks.values():
        m.close()

    detected = sum(f["detected"] for f in frames)
    data = {
        "source": video_path.name,
        "fps": fps,
        "width": width,
        "height": height,
        "crop": [x0, y0, x1, y1],
        "joints": list(JOINTS),
        "frames": frames,
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{video_path.name} -> {out_path.name}: {len(frames)} frames, detected {detected} "
          f"({detected / max(len(frames), 1):.0%}), 採用した回転 {rot_count}")


if __name__ == "__main__":
    jobs = JOBS if len(sys.argv) < 3 else [(sys.argv[1], sys.argv[2])]
    for video, out in jobs:
        extract(ROOT / video, ROOT / out)

"""① 動画から3D骨格を抽出して JSON に保存する（NLF: Neural Localizer Fields, GPU）。

    python extract_poses.py            # OK.mp4 -> poses_expert.json, me.mp4 -> poses_my.json
    python extract_poses.py in.mp4 out.json

MediaPipe は立った姿勢が前提で、ブレイクダンスの逆さ・回転姿勢では骨格が崩れるため、
多様な姿勢で学習された NLF (https://github.com/isarandi/nlf) で SMPL の24関節を推定する。

出力 JSON の各フレーム:
  world : カメラ座標系の3D位置 [m] (x右 y下 z奥)。腰中心ではなくカメラからの絶対位置
  image : 画像上の正規化座標 x,y (0-1)
  pose  : SMPL の関節回転 (軸角, 24x3)
複数人が映っている場合は、大きく映っていて前フレームの本人に近い人を追跡する（背景の人を拾わないように）。
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision  # noqa: F401  NLF 内の検出器が torchvision.ops.nms を使うので先に読み込む

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "nlf_l_multi_0.3.2.patch4.torchscript"
BATCH = 16
NLF_OPTIONS = dict(
    num_aug=5,                         # 回転させた5通りで推定して平均（安定化）
    detector_flip_aug=True,            # 人物検出も反転画像で行い、逆さの人を見逃しにくくする
    detector_both_flip_aug=True,
    suppress_implausible_poses=False,  # 逆さなど「ありえない姿勢」扱いで捨てられないように
)

# SMPL 24関節 -> 名前（既存の解析コードと同じ呼び名。nose は頭で代用）
SMPL_JOINTS = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
    "spine3", "l_foot_index", "r_foot_index", "neck", "l_collar", "r_collar", "head",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand",
]
EXTRA = {"nose": "head"}

JOBS = [("OK.mp4", "poses_expert.json"), ("me.mp4", "poses_my.json")]


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


def read_frames(video_path, box):
    x0, y0, x1, y1 = box
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return frames, fps


def pick_person(boxes, prev_box):
    """boxes: [N,5] (x,y,w,h,score)。背景の小さく映った人ではなく踊っている本人を選ぶ:
    大きさ（面積）を重視しつつ、前フレームの本人から離れた人は減点する"""
    if len(boxes) == 0:
        return None
    area = boxes[:, 2] * boxes[:, 3]
    if prev_box is None:
        return int(np.argmax(area))
    centers = boxes[:, :2] + boxes[:, 2:4] / 2
    prev_c = prev_box[:2] + prev_box[2:4] / 2
    size = np.sqrt(prev_box[2] * prev_box[3])
    dist = np.linalg.norm(centers - prev_c, axis=1) / size
    return int(np.argmax(area / (1.0 + dist ** 2)))


def extract(model, video_path: Path, out_path: Path) -> None:
    box = content_box(video_path)
    frames, fps = read_frames(video_path, box)
    H, W = frames[0].shape[:2]
    out = []
    prev_box = None
    for s in range(0, len(frames), BATCH):
        batch = torch.from_numpy(np.stack(frames[s:s + BATCH])).permute(0, 3, 1, 2).cuda()
        with torch.inference_mode(), torch.device("cuda"):
            pred = model.detect_smpl_batched(batch, **NLF_OPTIONS)
        for k in range(len(batch)):
            idx = s + k
            boxes = pred["boxes"][k].cpu().numpy() if "boxes" in pred else np.zeros((0, 5))
            i = pick_person(boxes, prev_box)
            frame = {"frame": idx, "time": round(idx / fps, 4), "detected": i is not None}
            if i is not None:
                prev_box = boxes[i]
                j3 = pred["joints3d"][k][i].cpu().numpy() / 1000.0  # mm -> m
                j2 = pred["joints2d"][k][i].cpu().numpy()
                frame["world"] = {n: [round(float(v), 5) for v in j3[t]] for t, n in enumerate(SMPL_JOINTS)}
                frame["image"] = {n: [round(float(j2[t, 0] / W), 5), round(float(j2[t, 1] / H), 5), 0.0]
                                  for t, n in enumerate(SMPL_JOINTS)}
                for alias, src in EXTRA.items():
                    frame["world"][alias] = frame["world"][src]
                    frame["image"][alias] = frame["image"][src]
                frame["visibility"] = {n: 1.0 for n in frame["world"]}
                frame["pose"] = [round(float(v), 5) for v in pred["pose"][k][i].cpu().numpy().ravel()]
            out.append(frame)
        print(f"  {video_path.name}: {min(s + BATCH, len(frames))}/{len(frames)}", flush=True)

    detected = sum(f["detected"] for f in out)
    data = {
        "source": video_path.name,
        "engine": "nlf",
        "fps": fps,
        "width": W,
        "height": H,
        "crop": list(box),
        "joints": SMPL_JOINTS + list(EXTRA),
        "frames": out,
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{video_path.name} -> {out_path.name}: {len(out)} frames, detected {detected} "
          f"({detected / max(len(out), 1):.0%})")


if __name__ == "__main__":
    with open(MODEL, "rb") as f:  # 日本語パスは torch.jit.load(パス) で開けないのでファイルオブジェクトで渡す
        model = torch.jit.load(f).cuda().eval()
    jobs = JOBS if len(sys.argv) < 3 else [(sys.argv[1], sys.argv[2])]
    for video, out in jobs:
        extract(model, ROOT / video, ROOT / out)

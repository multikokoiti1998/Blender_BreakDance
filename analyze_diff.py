"""② 自分(me.mp4)と見本(OK.mp4)の姿勢差分を算出し、ズレが大きいタイミング上位5箇所をレポートする。

2つの動画は長さも撮影アングルも違い、同じ時刻同士の比較は意味がないため:
  1. 視点に依存しにくい関節角度・体軸の傾き/ねじれを特徴量にする (pose_features.py)
  2. 自分の技の動き全体を、順序を保ったまま見本の技区間と部分DTWで照合
     （テンポ差 0.5〜2倍、左右反転も許容。見本に無い入り/抜けの動作は自動で除外）
  3. 対応するフレーム同士の差から、ズレのピーク上位5箇所を出力

    python analyze_diff.py
出力: report.txt, alignment.json (自分のフレーム -> 対応する見本フレーム。Blender 同期再生用)
"""
import json

import numpy as np

from pose_features import FEATURES, ROOT, compute_features, interp_smooth, load

TOP_N = 5
MIN_GAP_SEC = 0.2        # 上位ピーク同士の最小間隔
STALL_COST = 8.0         # DTW で見本側が止まる(自分の複数フレームが見本の1フレームに対応)ときのペナルティ [deg]
SUPPORT_JOINTS = ("l_wrist", "r_wrist", "l_hand", "r_hand")  # これが一番低い = 手で体を支えている
FOOT_JOINTS = ("l_ankle", "r_ankle", "l_foot_index", "r_foot_index")
ACTIVE_TILT = 40.0       # 体軸が鉛直からこれ以上傾いていれば「技に入っている」とみなす
MIN_VIS = 0.5            # 主要関節の平均 visibility の下限

LABELS_JA = {
    "torso_tilt": "体軸（腰→胸）の傾き",
    "axis_twist": "肩と腰のねじれ",
    "l_elbow": "左肘の伸び", "r_elbow": "右肘の伸び",
    "l_shoulder": "左腕の開き（体幹との角度）", "r_shoulder": "右腕の開き（体幹との角度）",
    "l_hip": "左股関節の伸び（体幹と太もも）", "r_hip": "右股関節の伸び（体幹と太もも）",
    "l_knee": "左膝の伸び", "r_knee": "右膝の伸び",
    "leg_split": "両脚の開き",
}
ADVICE_JA = {
    "torso_tilt": ("体軸が起きすぎ（腰が落ちている）", "体軸が倒れすぎ"),
    "axis_twist": ("肩と腰のねじれが小さい", "肩と腰がねじれすぎ（軸がブレている）"),
    "elbow": ("肘が曲がっている（ロックできていない）", "肘が伸びすぎ"),
    "shoulder": ("腕が体に近い（支えが狭い）", "腕が開きすぎ"),
    "hip": ("股関節が曲がっている（腰が折れている）", "股関節が反りすぎ"),
    "knee": ("膝が曲がっている", "膝が伸びすぎ"),
    "leg_split": ("脚の開きが足りない", "脚が開きすぎ"),
}

# 左右反転用: 特徴量の l_/r_ を入れ替えるインデックス
MIRROR = [FEATURES.index(f.replace("l_", "#").replace("r_", "l_").replace("#", "r_"))
          if f[:2] in ("l_", "r_") else i for i, f in enumerate(FEATURES)]


def segments(mask):
    """bool 配列の True 連続区間 [(start, end_exclusive), ...]"""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def estimate_period(F, fps):
    """体軸傾きの自己相関から1回転の周期(フレーム数)を推定"""
    x = F[:, FEATURES.index("torso_tilt")]
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo, hi = int(0.4 * fps), int(2.0 * fps)
    return lo + int(np.argmax(ac[lo:hi]))


def close_gaps(mask, max_gap):
    """max_gap フレーム以下の False の穴を埋める（単発の検出失敗で区間が切れないように）"""
    mask = mask.copy()
    for s, e in segments(~mask):
        if e - s <= max_gap and s > 0 and e < len(mask):
            mask[s:e] = True
    return mask


def subseq_dtw(W, E):
    """W[L,F] を E[N,F] の任意の部分区間と DTW 照合（W 側は全フレーム必須）。
    ステップ: 見本側を 0/1/2 フレーム進める = テンポ 0.5〜2倍を許容。
    戻り値: (平均コスト, W 各フレームに対応する E のインデックス)"""
    L, N = len(W), len(E)
    C = np.abs(W[:, None, :] - E[None, :, :]).mean(-1)  # [L,N] 平均角度差
    D = np.full((L, N), np.inf)
    P = np.zeros((L, N), dtype=np.int8)
    D[0] = C[0]
    for i in range(1, L):
        prev = D[i - 1]
        cand = np.stack([prev + STALL_COST,                      # 見本側 0 進む（止まり続けないようペナルティ）
                         np.concatenate([[np.inf], prev[:-1]]),  # 1 進む
                         np.concatenate([[np.inf, np.inf], prev[:-2]])])  # 2 進む
        P[i] = np.argmin(cand, axis=0)
        D[i] = C[i] + cand[P[i], np.arange(N)]
    j = int(np.argmin(D[-1]))
    cost = D[-1, j] / L
    path = [j]
    for i in range(L - 1, 0, -1):
        j -= int(P[i, j])
        path.append(j)
    return cost, path[::-1]


def active_mask(F, det, vis, fps):
    tilt = F[:, FEATURES.index("torso_tilt")]
    return close_gaps(det & (vis >= MIN_VIS) & (tilt >= ACTIVE_TILT), int(0.3 * fps))


def mean_visibility(name):
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return np.array([np.mean(list(f["visibility"].values())) if f["detected"] else 0.0
                     for f in data["frames"]])


def fmt_t(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def hip_lift(data, world):
    """腰の床からの高さ ÷ 胴の長さ（一番低い関節を床とみなす）"""
    ix = {j: i for i, j in enumerate(data["joints"])}
    pel = (world[:, ix["l_hip"]] + world[:, ix["r_hip"]]) / 2
    ch = (world[:, ix["l_shoulder"]] + world[:, ix["r_shoulder"]]) / 2
    torso = np.linalg.norm(ch - pel, axis=-1)
    floor = np.nanmax(world[:, :, 1], axis=1)  # y は下向き
    return interp_smooth(((floor - pel[:, 1]) / torso)[:, None])[:, 0]


def lowest_joint(data, world):
    """各フレームで一番低い（床に近い）関節名。未検出は None"""
    names = data["joints"]
    out = []
    for t in range(len(world)):
        y = world[t, :, 1]
        out.append(None if np.isnan(y).all() else names[int(np.nanargmax(y))])
    return out


def support_range(low, fps):
    """手で体を支えている最初〜最後のフレーム [s, e)。入りのステップや立ち上がりを除外する"""
    idx = [t for t, j in enumerate(low) if j in SUPPORT_JOINTS]
    return idx[0], idx[-1] + 1


def align_attempt(F_my, s, e, F_ex, ex_segs):
    """自分の [s, e) を、見本の技区間のどこか一続きの部分と順序を保って DTW 照合（左右反転も試す）"""
    best = None
    for mir in (False, True):
        W = F_my[s:e][:, MIRROR] if mir else F_my[s:e]
        for es, ee in ex_segs:
            cost, path = subseq_dtw(W, F_ex[es:ee])
            if best is None or cost < best[0]:
                best = (cost, mir, [es + p for p in path])
    return best


def main():
    d_ex, w_ex, _, det_ex = load("poses_expert.json")
    d_my, w_my, _, det_my = load("poses_my.json")
    fps_ex, fps_my = d_ex["fps"], d_my["fps"]
    F_ex = interp_smooth(compute_features(d_ex, w_ex))
    F_my = interp_smooth(compute_features(d_my, w_my))
    vis_ex = mean_visibility("poses_expert.json")
    vis_my = mean_visibility("poses_my.json")
    lift_ex, lift_my = hip_lift(d_ex, w_ex), hip_lift(d_my, w_my)

    # 見本の技区間（1秒未満の短い区間は除外）
    act_ex = active_mask(F_ex, det_ex, vis_ex, fps_ex)
    ex_segs = [(s, e) for s, e in segments(act_ex) if e - s >= fps_ex]
    period = estimate_period(F_ex[ex_segs[0][0]:ex_segs[0][1]], fps_ex)
    act_my = active_mask(F_my, det_my, vis_my, fps_my)

    low_my = lowest_joint(d_my, w_my)
    s, e = support_range(low_my, fps_my)
    mean_cost, mir, path = align_attempt(F_my, s, e, F_ex, ex_segs)
    T = len(F_my)
    # 手以外（膝・腰・肩など）が一番低い = 体が床に着いている
    touch = [t for t in range(s, e) if low_my[t] not in SUPPORT_JOINTS and low_my[t] not in FOOT_JOINTS]
    match = np.full(T, -1)
    match[s:e] = path
    mine = F_my[:, MIRROR] if mir else F_my
    frame_diff = np.full(T, np.nan)
    frame_diff[s:e] = np.abs(mine[s:e] - F_ex[path]).mean(1)
    print(f"照合: 自分 {s / fps_my:.2f}-{e / fps_my:.2f}s  ⇔  見本 {path[0] / fps_ex:.2f}-{path[-1] / fps_ex:.2f}s"
          f"  平均ズレ {mean_cost:.1f}°  左右反転={mir}")

    # ピーク抽出
    order = np.argsort(np.nan_to_num(frame_diff, nan=-1))[::-1]
    peaks = []
    for c in order:
        if np.isnan(frame_diff[c]):
            break
        if all(abs(c - p) >= MIN_GAP_SEC * fps_my for p in peaks):
            peaks.append(int(c))
        if len(peaks) == TOP_N:
            break

    lines = [
        f"ブレイクダンス姿勢比較レポート（自分: {d_my['source']} / 見本: {d_ex['source']}）",
        "=" * 64,
        f"比較した区間: 自分 {fmt_t(s / fps_my)}〜{fmt_t(e / fps_my)}  ⇔  見本 {fmt_t(path[0] / fps_ex)}〜{fmt_t(path[-1] / fps_ex)}"
        + ("  （左右反転して照合）" if mir else ""),
        f"  （手で体を支えている最初〜最後のフレーム。入りのステップ等は見本に無いので比較対象外）",
        f"見本の1回転周期: {period / fps_ex:.2f} 秒 / 自分の比較区間の長さ: {(e - s) / fps_my:.2f} 秒"
        f"（見本では {(path[-1] - path[0] + 1) / fps_ex:.2f} 秒ぶんの動き）",
        f"比較区間の平均ズレ: {mean_cost:.1f}°",
        "",
        "【床に体が着いた瞬間】（手・足以外の関節が一番低い位置にある）",
        *(["   " + ", ".join(f"{fmt_t(a / fps_my)}〜{fmt_t(b / fps_my)}（{low_my[a]}）" for a, b in segments(
            np.isin(np.arange(T), touch)))] if touch else ["   なし"]),
        "",
        "※ 自分の技の動き全体を、順序を保ったまま見本の動きとDTWで対応付け（テンポ差は吸収）、",
        "   対応するフレーム同士で関節角度の差が大きい順に並べています。",
        "※ 角度は 180°=まっすぐ伸びている。体軸の傾きは 0°=直立 / 90°=水平 / 180°=逆さ。",
        "※ 腰の高さ = 床（一番低い関節）から腰までの高さ ÷ 胴の長さ。",
        "",
    ]
    for rank, c in enumerate(peaks, 1):
        ex = int(match[c])
        diff = mine[c] - F_ex[ex]
        lines.append(f"【{rank}位】 自分 {fmt_t(c / fps_my)} (frame {c})  ⇔  見本 {fmt_t(ex / fps_ex)} (frame {ex})"
                     f"   ズレ {frame_diff[c]:.1f}°")
        lines.append(f"   - 腰の高さ: 自分 {lift_my[c]:.2f} / 見本 {lift_ex[ex]:.2f}")
        for k in np.argsort(-np.abs(diff))[:4]:
            name = FEATURES[k]
            key = name[2:] if name[:2] in ("l_", "r_") else name
            adv = ADVICE_JA[key][0 if diff[k] < 0 else 1]
            lines.append(f"   - {LABELS_JA[name]}: 自分 {mine[c, k]:5.1f}° / 見本 {F_ex[ex, k]:5.1f}° "
                         f"(差 {diff[k]:+6.1f}°) → {adv}")
        lines.append("")

    # 全体傾向: 比較区間全体での平均差（符号付き）
    diffs = mine[s:e] - F_ex[path]
    lines.append("【全体傾向】比較区間全体での平均差（自分 − 見本）")
    lines.append(f"   {'腰の高さ':<24} {np.mean(lift_my[s:e] - lift_ex[path]):+6.2f}"
                 f"  (自分 平均 {lift_my[s:e].mean():.2f} / 見本 {lift_ex[path].mean():.2f})")
    for k in np.argsort(-np.abs(diffs.mean(0))):
        lines.append(f"   {LABELS_JA[FEATURES[k]]:<24} {diffs[:, k].mean():+6.1f}°")

    report = "\n".join(lines)
    (ROOT / "report.txt").write_text(report, encoding="utf-8")
    print(report)

    # Blender 同期用: 比較区間外は端の対応を延長
    m = np.clip(np.arange(T), s, e - 1)
    (ROOT / "alignment.json").write_text(json.dumps({
        "my_fps": fps_my, "expert_fps": fps_ex,
        "my_to_expert": [int(match[i]) for i in m],
        "score": [None if np.isnan(x) else round(float(x), 2) for x in frame_diff],
        "mirrored": bool(mir),
        "compared_range": [int(s), int(e)],
        "floor_touch": touch,
        "top_frames": peaks,
        # Blender の軌跡・配置に使う区間 = 実際に比較した区間
        "my_active": [bool(s <= t < e) for t in range(T)],
        "expert_active": [bool(path[0] <= t <= path[-1]) for t in range(len(F_ex))],
    }), encoding="utf-8")


if __name__ == "__main__":
    main()

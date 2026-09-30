"""③ poses_my.json / poses_expert.json を Blender の2体の人型リグに適用し、腰(Pelvis)の軌跡を描画する。

使い方:
  A) Blender の Scripting タブでこのファイルを開いて「Run Script」
  B) コマンドライン:
     blender -P apply_poses.py                       # GUI で開く
     blender -b -P apply_poses.py -- --save out.blend  # 生成して .blend 保存

リグ:
  このフォルダに Mixamo の Y-Bot (*.fbx, 例: "Y Bot.fbx") があればそれを2体読み込んで使う。
  無ければ関節をつないだスティック人形リグを自動生成する。

仕組み:
  関節ごとに Empty を作ってフレーム毎の座標をキーフレーム。
  スティックリグはコンストレイントで Empty を追従、Mixamo リグは各部位の向きを回転キーフレーム(FK)として焼き込む
  （FK なので Blender 上でポーズを手で修正できる）。
  自分 = 赤 (左側), 見本 = 青 (右側)。タイムラインは自分の動画基準で、
  ALIGN=True なら見本は analyze_diff.py の DTW 照合結果に合わせて同期再生される。
  ズレ上位5箇所はタイムラインマーカー (#1〜#5) として入る。
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

# ---------------- 設定 ----------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path(bpy.path.abspath("//"))
if not (ROOT / "poses_my.json").exists():  # Scripting タブで実行すると __file__ がテキスト名になる場合
    ROOT = Path(bpy.path.abspath("//")) if bpy.data.filepath else ROOT
    for t in bpy.data.texts:
        if t.filepath and (Path(bpy.path.abspath(t.filepath)).parent / "poses_my.json").exists():
            ROOT = Path(bpy.path.abspath(t.filepath)).parent

ALIGN = True          # True: 見本を DTW 照合で自分のタイミングに同期 / False: それぞれ元のタイミング
SPACING = 0.8         # 2体の移動範囲どうしの隙間 [m]
SMOOTH_WIN = 3        # 座標の移動平均フレーム数（大きいと速い動きが鈍る）
COLORS = {"my": (0.9, 0.1, 0.1, 1.0), "expert": (0.1, 0.3, 0.95, 1.0)}

# extract_poses.py (NLF / SMPL 24関節) の関節名
JOINTS = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
          "spine3", "l_foot_index", "r_foot_index", "neck", "l_collar", "r_collar", "head",
          "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand", "nose"]
TARGETS = JOINTS + ["chest"]

# スティックリグ: (ボーン名, 根元, 先端)
STICK_BONES = [
    ("spine", "pelvis", "neck"), ("neck", "neck", "head"),
    ("hip.L", "pelvis", "l_hip"), ("hip.R", "pelvis", "r_hip"),
    ("clavicle.L", "neck", "l_shoulder"), ("clavicle.R", "neck", "r_shoulder"),
    ("hand.L", "l_wrist", "l_hand"), ("hand.R", "r_wrist", "r_hand"),
    ("foot.L", "l_ankle", "l_foot_index"), ("foot.R", "r_ankle", "r_foot_index"),
    ("upper_arm.L", "l_shoulder", "l_elbow"), ("forearm.L", "l_elbow", "l_wrist"),
    ("upper_arm.R", "r_shoulder", "r_elbow"), ("forearm.R", "r_elbow", "r_wrist"),
    ("thigh.L", "l_hip", "l_knee"), ("shin.L", "l_knee", "l_ankle"),
    ("thigh.R", "r_hip", "r_knee"), ("shin.R", "r_knee", "r_ankle"),
]

# ---------------- データ前処理 ----------------
def interp_smooth(x, win):
    """x[T,...] の NaN を時間方向に補間して移動平均"""
    shp = x.shape
    x = x.reshape(len(x), -1).copy()
    t = np.arange(len(x))
    for c in range(x.shape[1]):
        ok = ~np.isnan(x[:, c])
        if ok.sum() >= 2:
            x[:, c] = np.interp(t, t[ok], x[ok, c])
        else:
            x[:, c] = 0.0
    if win > 1:
        k = np.ones(win) / win
        pad = np.pad(x, ((win // 2, win // 2), (0, 0)), mode="edge")
        x = np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(x.shape[1])], axis=1)
    return x.reshape(shp)


def rotation_to_z(v):
    """ベクトル v を +Z に向ける回転行列"""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s, c = np.linalg.norm(axis), float(np.dot(v, z))
    if s < 1e-8:
        return np.eye(3)
    axis /= s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + K * s + K @ K * (1 - c)


def estimate_floor(P, active):
    """床の向き(上方向)と高さを推定し、床を Z=0 の水平面にする回転 R と高さを返す。
    1) 立っているフレームがあれば体軸を鉛直とみなす
    2) 無ければ、各フレームの一番低い関節（床に接している点）に平面を当てはめる"""
    ti = {n: i for i, n in enumerate(TARGETS)}
    act = np.array(active, bool) & ~np.isnan(P[:, 0, 0])
    spine = P[:, ti["neck"]] - P[:, ti["pelvis"]]
    n = spine / np.linalg.norm(spine, axis=-1, keepdims=True)
    standing = n[:, 2] > np.cos(np.radians(20))
    if standing.sum() >= 10:
        R = rotation_to_z(np.nanmean(n[standing], axis=0))
    else:
        low = np.array([P[t, np.nanargmin(P[t, :, 2])] for t in np.where(act)[0]])
        keep = np.ones(len(low), bool)
        coef = np.zeros(3)
        for _ in range(3):  # 浮いている（床に着いていない）フレームを除きながら Z = aX + bY + c を当てはめ
            A = np.c_[low[keep, :2], np.ones(keep.sum())]
            lam = 0.5 * keep.sum() * np.diag([1, 1, 0])  # 傾きは小さい方へ寄せる（点が一直線に並ぶ時の暴れ防止）
            coef = np.linalg.solve(A.T @ A + lam, A.T @ low[keep, 2])
            resid = low[:, 2] - (low[:, :2] @ coef[:2] + coef[2])
            keep = resid < np.percentile(resid, 60) + 0.03
        normal = np.array([-coef[0], -coef[1], 1.0])
        normal /= np.linalg.norm(normal)
        R = rotation_to_z(normal) if normal[2] > np.cos(np.radians(35)) else np.eye(3)
    Q = P @ R.T
    lows = np.nanmin(Q[act][:, :, 2], axis=1)
    floor = np.percentile(lows, 20)  # 手足が床に着いている高さ（浮いている時間は除く）
    return R, floor


def load_positions(name, active):
    """JSON -> Blender 座標系の関節位置 P[T, len(TARGETS), 3] (床=Z0、カメラからの実際の位置)"""
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    T = len(data["frames"])
    world = np.full((T, len(JOINTS), 3), np.nan)
    for t, f in enumerate(data["frames"]):
        if f["detected"]:
            world[t] = [f["world"][j] for j in JOINTS]
    # カメラ座標 (x右, y下, z奥) -> Blender (X右, Y奥, Z上)
    P = np.stack([world[..., 0], world[..., 2], -world[..., 1]], axis=-1)
    ix = {j: i for i, j in enumerate(JOINTS)}
    chest = (P[:, ix["l_shoulder"]] + P[:, ix["r_shoulder"]]) / 2
    P = interp_smooth(np.concatenate([P, chest[:, None]], axis=1), SMOOTH_WIN)

    R, floor = estimate_floor(P, active)
    P = P @ R.T
    P[:, :, 2] -= floor
    # 比較区間の腰の平均位置を原点に
    act = np.array(active, bool)
    P[:, :, :2] -= np.median(P[act, ix["pelvis"], :2], axis=0)
    return P, data["fps"]


# ---------------- Blender ユーティリティ ----------------
def fcurves_of(obj):
    """Blender 4.4+ (スロット付きアクション) と旧 API の両方で fcurves を返す"""
    act = obj.animation_data.action
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        return action_get_channelbag_for_slot(act, obj.animation_data.action_slot).fcurves
    except ImportError:
        return act.fcurves


def bake_locations(obj, locs, frame_offset=1):
    """obj.location を全フレーム一括キーフレーム (locs[T,3])"""
    obj.location = locs[0]
    obj.keyframe_insert("location", frame=frame_offset)
    frames = np.arange(len(locs), dtype=np.float32) + frame_offset
    for fc in fcurves_of(obj):
        if fc.data_path != "location":
            continue
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(locs))
        co = np.empty(len(locs) * 2, np.float32)
        co[0::2], co[1::2] = frames, locs[:, fc.array_index]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [1] * len(locs))  # LINEAR
        fc.update()


def material(name, rgba, emission=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        if emission:
            bsdf.inputs["Emission Color"].default_value = rgba
            bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def new_collection(name):
    old = bpy.data.collections.get(name)
    if old:
        for o in list(old.all_objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def link(obj, col):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)


def make_empties(P, key, col, offset, frame_map=None):
    """関節ごとの Empty を作ってキーフレーム。frame_map があれば見本フレームを自分のタイムラインに並べ替え"""
    if frame_map is not None:
        P = P[np.clip(frame_map, 0, len(P) - 1)]
    empties = {}
    for i, name in enumerate(TARGETS):
        e = bpy.data.objects.new(f"{key}_{name}", None)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = 0.03
        col.objects.link(e)
        bake_locations(e, P[:, i] + offset)
        empties[name] = e
    return empties, P + offset


def build_stick_rig(key, empties, col):
    arm_data = bpy.data.armatures.new(f"{key}_rig")
    arm = bpy.data.objects.new(f"{key}_rig", arm_data)
    col.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for bname, a, b in STICK_BONES:
        eb = arm_data.edit_bones.new(bname)
        eb.head, eb.tail = (0, 0, 0), (0, 0.1, 0)  # 実際の長さは Stretch To が決める
    bpy.ops.object.mode_set(mode="POSE")
    mat = material(f"{key}_body", COLORS[key])
    for bname, a, b in STICK_BONES:
        pb = arm.pose.bones[bname]
        c = pb.constraints.new("COPY_LOCATION")
        c.target = empties[a]
        c = pb.constraints.new("STRETCH_TO")
        c.target = empties[b]
        c.volume = "NO_VOLUME"
        c.rest_length = 0.1
        # 見た目用の円柱をボーンに親子付け
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.035, depth=0.1,
                                            location=(0, 0.05, 0), rotation=(np.pi / 2, 0, 0))
        cyl = bpy.context.active_object
        cyl.name = f"{key}_{bname}_mesh"
        cyl.data.materials.append(mat)
        link(cyl, col)
        cyl.parent, cyl.parent_type, cyl.parent_bone = arm, "BONE", bname
        cyl.matrix_parent_inverse = Matrix.Translation((0, -0.1, 0))  # ボーン先端基準 -> 根元基準
        bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="OBJECT")
    # 頭
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, segments=16, ring_count=8)
    head = bpy.context.active_object
    head.name = f"{key}_head"
    head.data.materials.append(mat)
    link(head, col)
    c = head.constraints.new("COPY_LOCATION")
    c.target = empties["head"]
    arm.show_in_front = True
    arm.data.display_type = "STICK"
    return arm


def find_mixamo_fbx():
    fbx = sorted(ROOT.glob("*.fbx")) + sorted((ROOT / "models").glob("*.fbx"))
    for p in fbx:
        if "bot" in p.stem.lower() or "mixamo" in p.stem.lower():
            return p
    return fbx[0] if fbx else None


def build_mixamo_rig(key, P, col, fbx):
    """Mixamo リグに FK で回転キーフレームを焼き込む。
    Y-Bot と本人では手足の長さが違うので、関節の「位置」ではなく各部位の「向き」を写す。
    P: [T, len(TARGETS), 3] ワールド座標（配置オフセット込み）"""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx), automatic_bone_orientation=False)
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        link(o, col)
    arm = next(o for o in new if o.type == "ARMATURE")
    arm.name = f"{key}_rig"
    if arm.animation_data:
        arm.animation_data.action = None  # FBX 付属の T ポーズアクションを外す
    mat = material(f"{key}_body", COLORS[key])
    joint_mat = material(f"{key}_joints", tuple(c * 0.4 for c in COLORS[key][:3]) + (1.0,))
    for o in new:
        if o.type == "MESH":
            o.data.materials.clear()
            o.data.materials.append(joint_mat if "joint" in o.name.lower() else mat)
    prefix = next((b.name.split("Hips")[0] for b in arm.data.bones if b.name.endswith("Hips")), "mixamorig:")
    bones = arm.data.bones
    B = lambda n: bones.get(prefix + n)
    ti = {n: i for i, n in enumerate(TARGETS)}

    # 脚の長さを本人に合わせてスケール
    def leg_len_rig():
        s = arm.matrix_world.to_scale()[0]
        return ((B("LeftLeg").head_local - B("LeftUpLeg").head_local).length
                + (B("LeftFoot").head_local - B("LeftLeg").head_local).length) * s
    leg_data = np.nanmedian(np.linalg.norm(P[:, ti["l_knee"]] - P[:, ti["l_hip"]], axis=-1)
                            + np.linalg.norm(P[:, ti["l_ankle"]] - P[:, ti["l_knee"]], axis=-1))
    arm.scale = arm.scale * (leg_data / leg_len_rig())
    bpy.context.view_layer.update()
    A_inv = arm.matrix_world.inverted()
    R_inv = A_inv.to_3x3()

    def side_sign(bone, left, right):
        x = bones[prefix + bone].matrix_local.to_3x3().col[0]
        return 1.0 if x.dot(B(left).head_local - B(right).head_local) > 0 else -1.0

    # ボーン -> (向きの始点, 終点, ねじれ用の左右ベクトル(始点,終点,符号) or None)
    drive = {
        "Hips": ("pelvis", "spine1", ("l_hip", "r_hip", side_sign("Hips", "LeftUpLeg", "RightUpLeg"))),
        "Spine": ("spine1", "spine2", None), "Spine1": ("spine2", "spine3", None),
        "Spine2": ("spine3", "neck", ("l_shoulder", "r_shoulder", side_sign("Spine2", "LeftArm", "RightArm"))),
        "Neck": ("neck", "head", None), "Head": ("neck", "head", None),
        "LeftShoulder": ("l_collar", "l_shoulder", None), "RightShoulder": ("r_collar", "r_shoulder", None),
        "LeftArm": ("l_shoulder", "l_elbow", None), "LeftForeArm": ("l_elbow", "l_wrist", None),
        "LeftHand": ("l_wrist", "l_hand", None),
        "RightArm": ("r_shoulder", "r_elbow", None), "RightForeArm": ("r_elbow", "r_wrist", None),
        "RightHand": ("r_wrist", "r_hand", None),
        "LeftUpLeg": ("l_hip", "l_knee", None), "LeftLeg": ("l_knee", "l_ankle", None),
        "LeftFoot": ("l_ankle", "l_foot_index", None),
        "RightUpLeg": ("r_hip", "r_knee", None), "RightLeg": ("r_knee", "r_ankle", None),
        "RightFoot": ("r_ankle", "r_foot_index", None),
    }
    order = sorted(bones, key=lambda b: len(b.parent_recursive))
    driven = [b for b in order if b.name[len(prefix):] in drive]
    T = len(P)
    quats = {b.name: np.zeros((T, 4)) for b in driven}
    hips_loc = np.zeros((T, 3))

    def adir(t, a, b):
        return (R_inv @ Vector(P[t, ti[b]] - P[t, ti[a]])).normalized()

    for t in range(T):
        pose = {}
        for b in order:
            rest = b.matrix_local
            carried = pose[b.parent.name] @ b.parent.matrix_local.inverted() @ rest if b.parent else rest.copy()
            spec = drive.get(b.name[len(prefix):])
            if spec is None:
                pose[b.name] = carried
                continue
            a, c, twist = spec
            d = adir(t, a, c)
            if twist:
                x = adir(t, twist[0], twist[1]) * twist[2]
                x = (x - d * x.dot(d)).normalized()
                rot = Matrix((x, d, x.cross(d))).transposed()
            else:
                rot = carried.to_3x3().col[1].normalized().rotation_difference(d).to_matrix() @ carried.to_3x3()
            head = A_inv @ Vector(P[t, ti["pelvis"]]) if b.parent is None else carried.translation
            desired = Matrix.LocRotScale(head, rot.to_quaternion(), None)
            pose[b.name] = desired
            basis = carried.inverted() @ desired
            q = basis.to_quaternion()
            if t and np.dot(quats[b.name][t - 1], q) < 0:  # クォータニオンの符号反転を防ぐ
                q.negate()
            quats[b.name][t] = q
            if b.parent is None:
                hips_loc[t] = basis.translation

    # キーフレームを一括で焼き込む
    for b in driven:
        pb = arm.pose.bones[b.name]
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=1)
        if b.parent is None:
            pb.keyframe_insert("location", frame=1)
    frames = np.arange(T, dtype=np.float32) + 1
    for fc in fcurves_of(arm):
        name = fc.data_path.split('"')[1]
        if fc.data_path.endswith("rotation_quaternion"):
            vals = quats[name][:, fc.array_index]
        elif fc.data_path.endswith("location"):
            vals = hips_loc[:, fc.array_index]
        else:
            continue
        fc.keyframe_points.clear()
        fc.keyframe_points.add(T)
        co = np.empty(T * 2, np.float32)
        co[0::2], co[1::2] = frames, vals
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [1] * T)
        fc.update()
    return arm


def make_trajectory(key, P, active, offset, col):
    """腰の軌跡カーブ (技に入っている区間ごとにスプライン分割)"""
    pel = P[:, TARGETS.index("pelvis")] + offset
    cu = bpy.data.curves.new(f"{key}_pelvis_path", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.008
    act = np.concatenate([[False], np.array(active, bool), [False]]).astype(int)
    starts, ends = np.where(np.diff(act) == 1)[0], np.where(np.diff(act) == -1)[0]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        sp = cu.splines.new("POLY")
        sp.points.add(e - s - 1)
        co = np.concatenate([pel[s:e], np.ones((e - s, 1))], axis=1).astype(np.float32)
        sp.points.foreach_set("co", co.ravel())
    obj = bpy.data.objects.new(f"{key}_pelvis_path", cu)
    obj.data.materials.append(material(f"{key}_path", COLORS[key], emission=2.0))
    obj.color = COLORS[key]
    col.objects.link(obj)
    return obj


# ---------------- メイン ----------------
def main():
    scene = bpy.context.scene
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for name in ("Cube",):  # 起動時の既定キューブを削除
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    al = json.loads((ROOT / "alignment.json").read_text(encoding="utf-8"))
    P_my, fps = load_positions("poses_my.json", al["my_active"])
    P_ex, _ = load_positions("poses_expert.json", al["expert_active"])
    frame_map = np.array(al["my_to_expert"]) if ALIGN else None

    T = len(P_my) if ALIGN else max(len(P_my), len(P_ex))
    scene.render.fps = int(round(fps))
    scene.frame_start, scene.frame_end = 1, T

    fbx = find_mixamo_fbx()
    print(f"[apply_poses] ROOT={ROOT} rig={'Mixamo: ' + fbx.name if fbx else 'stick'} ALIGN={ALIGN}")
    # 2体の移動範囲が重ならないよう、自分の右端と見本の左端の間に SPACING 空ける
    def x_range(P, active):
        xs = P[np.array(active, bool)][:, :, 0]
        return np.percentile(xs, 2), np.percentile(xs, 98)
    my_lo, my_hi = x_range(P_my, al["my_active"])
    ex_lo, ex_hi = x_range(P_ex, al["expert_active"])
    x_my = -my_hi - SPACING / 2
    x_ex = -ex_lo + SPACING / 2
    for key, P, active, x in (("my", P_my, al["my_active"], x_my),
                              ("expert", P_ex, al["expert_active"], x_ex)):
        col = new_collection(f"{key}_{'自分' if key == 'my' else '見本'}")
        offset = np.array([x, 0.0, 0.0])
        empties, P_world = make_empties(P, key, col, offset, frame_map if key == "expert" else None)
        if fbx:
            build_mixamo_rig(key, P_world, col, fbx)
        else:
            build_stick_rig(key, empties, col)
        make_trajectory(key, P, active, offset, col)

    # 床とカメラ（接地が分かるように）
    for n in ("floor", "compare_camera"):
        if n in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[n], do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "floor"
    floor.data.materials.append(material("floor", (0.55, 0.5, 0.45, 1.0)))
    cam = bpy.data.objects.new("compare_camera", bpy.data.cameras.new("compare_camera"))
    scene.collection.objects.link(cam)
    cam.location = ((x_my + x_ex) / 2, -5.0, 1.3)
    cam.rotation_euler = (np.radians(80), 0, 0)
    cam.data.lens = 30
    scene.camera = cam
    scene.display.shading.show_shadows = True
    scene.display.shading.color_type = "MATERIAL"

    # ズレ上位5箇所をマーカーに
    for m in list(scene.timeline_markers):
        if m.name.startswith("#"):
            scene.timeline_markers.remove(m)
    for rank, f in enumerate(al["top_frames"], 1):
        scene.timeline_markers.new(f"#{rank} ズレ{al['score'][f]:.0f}°", frame=f + 1)

    # ソリッド表示でオブジェクト色が見えるように
    for area in (bpy.context.screen.areas if bpy.context.screen else []):
        if area.type == "VIEW_3D":
            area.spaces[0].shading.color_type = "MATERIAL"
    scene.frame_set(al["top_frames"][0] + 1 if al["top_frames"] else 1)
    print("[apply_poses] done")


main()

if "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1:]
    if "--save" in args:
        out = Path(args[args.index("--save") + 1])
        bpy.ops.wm.save_as_mainfile(filepath=str(out if out.is_absolute() else ROOT / out))

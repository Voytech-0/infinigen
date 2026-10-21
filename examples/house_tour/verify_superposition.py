"""Independent verification of a house_tour superposition blend.

Usage:
    uv run python verify_superposition.py path/to/scene_superposition.blend [--max_changes N]

Checks (each prints PASS/FAIL; exit code 1 on any FAIL):
 1. structure   - hide fcurves on every unit descendant, CONSTANT keys, no light near origin
 2. unseen      - every unit state change is invisible from the camera at f-1 and f
                  (scene.ray_cast on the real depsgraph, AABB surface sampling)
 3. variation   - some unit has >=2 distinct states, some unit goes absent -> present
 4. summary     - unit / change counts per room

Does not import anything from infinigen; reads only the saved blend.
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector

UNIT_PREFIX = "quantum_unit_"
ROOM_PREFIX = "house_room_"
HIDE_PATHS = ("hide_render", "hide_viewport")
EMPTY_PATHS = ("location", "rotation_euler")
GRID_SPACING = 0.2
SHRINK = 0.98
HIT_TOL = 0.1
LIGHT_ORIGIN_RADIUS = 0.5
MARGINAL_POINTS = 2
BBOX_TYPES = {
    "MESH",
    "CURVE",
    "SURFACE",
    "META",
    "FONT",
    "CURVES",
    "POINTCLOUD",
    "VOLUME",
}

results: list[tuple[str, bool]] = []


def report(name: str, details: list[str], ok: bool):
    results.append((name, ok))
    print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
    for line in details:
        print(f"    {line}")


def fcurve(obj, path, index=0):
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return None
    return ad.action.fcurves.find(path, index=index)


def unhide_rooms(scene):
    rooms = [c for c in bpy.data.collections if c.name.startswith(ROOM_PREFIX)]
    for c in rooms:
        c.hide_viewport = False

    def walk(lc):
        if lc.collection.name.startswith(ROOM_PREFIX):
            lc.hide_viewport = False
            lc.exclude = False
        for ch in lc.children:
            walk(ch)

    for vl in scene.view_layers:
        walk(vl.layer_collection)
    bpy.context.view_layer.update()
    return rooms


def unit_room(empty):
    for c in empty.users_collection:
        if c.name.startswith(ROOM_PREFIX):
            return c.name
    return "<no room>"


def unit_state(empty, members, frame):
    """State tuple from fcurve evaluation (no frame_set)."""
    loc = []
    for i in range(3):
        fc = fcurve(empty, "location", i)
        loc.append(round(fc.evaluate(frame), 4) if fc else round(empty.location[i], 4))
    rot = []
    for i in range(3):
        fc = fcurve(empty, "rotation_euler", i)
        rot.append(
            round(fc.evaluate(frame), 4) if fc else round(empty.rotation_euler[i], 4)
        )
    hidden = []
    for m in members:
        fc = fcurve(m, "hide_render")
        hidden.append(bool(fc.evaluate(frame) > 0.5) if fc else bool(m.hide_render))
    return tuple(loc), tuple(rot), tuple(hidden)


def key_frames(empty, members):
    frames = set()
    for path in EMPTY_PATHS:
        for i in range(3):
            fc = fcurve(empty, path, i)
            if fc:
                frames.update(round(k.co[0]) for k in fc.keyframe_points)
    for m in members:
        for path in HIDE_PATHS:
            fc = fcurve(m, path)
            if fc:
                frames.update(round(k.co[0]) for k in fc.keyframe_points)
    return sorted(frames)


# ---------------------------------------------------------------- check 1


def check_structure(scene, units):
    details = []
    missing, nonconst = [], []
    for empty, members in units.items():
        for m in members:
            for path in HIDE_PATHS:
                fc = fcurve(m, path)
                if fc is None:
                    missing.append(f"{empty.name}/{m.name}: no {path} fcurve")
                elif any(k.interpolation != "CONSTANT" for k in fc.keyframe_points):
                    nonconst.append(f"{empty.name}/{m.name}.{path}")
        for path in EMPTY_PATHS:
            for i in range(3):
                fc = fcurve(empty, path, i)
                if fc and any(
                    k.interpolation != "CONSTANT" for k in fc.keyframe_points
                ):
                    nonconst.append(f"{empty.name}.{path}[{i}]")
        # parent chain sanity (members come from children_recursive, so this guards
        # against odd parenting like parent_type changes or cycles)
        names = {m.name for m in members} | {empty.name}
        for m in members:
            if m.parent is None or m.parent.name not in names:
                missing.append(f"{empty.name}/{m.name}: parent {m.parent} outside unit")
    scene.frame_set(scene.frame_start)
    near = [
        f"{o.name} @ frame {scene.frame_start}: {tuple(round(v, 3) for v in o.matrix_world.translation)}"
        for o in scene.objects
        if o.type == "LIGHT" and o.matrix_world.translation.length < LIGHT_ORIGIN_RADIUS
    ]
    n_members = sum(len(m) for m in units.values())
    details.append(f"{len(units)} units, {n_members} member/descendant objects")
    details.append(f"missing hide fcurves / bad parent: {len(missing)}")
    details += ["  " + s for s in missing[:20]]
    details.append(f"non-CONSTANT fcurves: {len(nonconst)}")
    details += ["  " + s for s in nonconst[:20]]
    details.append(f"lights within {LIGHT_ORIGIN_RADIUS} m of origin: {len(near)}")
    details += ["  " + s for s in near[:20]]
    return details, not (missing or nonconst or near)


# ---------------------------------------------------------------- check 2


def box_points(lo, hi):
    c = (lo + hi) / 2
    lo = c + (lo - c) * SHRINK
    hi = c + (hi - c) * SHRINK
    pts = []
    for a in range(3):
        b, d = [x for x in range(3) if x != a]
        nb = max(2, math.ceil((hi[b] - lo[b]) / GRID_SPACING) + 1)
        nd = max(2, math.ceil((hi[d] - lo[d]) / GRID_SPACING) + 1)
        gb, gd = np.meshgrid(
            np.linspace(lo[b], hi[b], nb), np.linspace(lo[d], hi[d], nd)
        )
        for side in (lo[a], hi[a]):
            p = np.zeros((gb.size, 3))
            p[:, a] = side
            p[:, b] = gb.ravel()
            p[:, d] = gd.ravel()
            pts.append(p)
    return np.unique(np.round(np.concatenate(pts), 5), axis=0)


def visible_box(depsgraph, members):
    """World AABB of currently visible members (evaluated)."""
    corners = []
    for m in members:
        if m.hide_render or m.type not in BBOX_TYPES:
            continue
        ev = m.evaluated_get(depsgraph)
        mw = ev.matrix_world
        corners += [mw @ Vector(c) for c in ev.bound_box]
    if not corners:
        return None
    arr = np.array([tuple(c) for c in corners])
    return arr.min(0), arr.max(0)


def camera_frustum(scene, cam):
    mw = cam.matrix_world.copy()
    inv = mw.inverted()
    frame = cam.data.view_frame(scene=scene)
    xs = [v.x / -v.z for v in frame]
    ys = [v.y / -v.z for v in frame]
    return mw.translation.copy(), inv, (min(xs), max(xs), min(ys), max(ys))


def count_visible(scene, depsgraph, cam_info, pts, own_names):
    cam_pos, inv, (x0, x1, y0, y1) = cam_info
    n_in = n_vis = 0
    for p in pts:
        pw = Vector(p)
        pc = inv @ pw
        if pc.z >= 0:
            continue
        sx, sy = pc.x / -pc.z, pc.y / -pc.z
        if not (x0 <= sx <= x1 and y0 <= sy <= y1):
            continue
        n_in += 1
        d = pw - cam_pos
        dist = d.length
        if dist < 1e-6:
            n_vis += 1
            continue
        hit, loc, _n, _i, obj, _m = scene.ray_cast(
            depsgraph, cam_pos, d / dist, distance=dist
        )
        if (
            not hit
            or (obj is not None and obj.name in own_names)
            or (loc - pw).length <= HIT_TOL
        ):
            n_vis += 1
    return n_in, n_vis


def check_unseen(scene, cam, units, changes):
    # pass 1: boxes (old @ f-1, new @ f)
    need_box = defaultdict(list)  # frame -> [(change_idx, 'old'|'new')]
    for i, (f, empty) in enumerate(changes):
        need_box[f - 1].append((i, "old"))
        need_box[f].append((i, "new"))
    boxes = {}
    t0 = time.time()
    for fr in sorted(need_box):
        scene.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        for i, kind in need_box[fr]:
            boxes[(i, kind)] = visible_box(dg, units[changes[i][1]])
    t1 = time.time()
    # pass 2: ray tests, each box at both f-1 and f
    need_test = defaultdict(list)
    for i, (f, _e) in enumerate(changes):
        for kind in ("old", "new"):
            if boxes[(i, kind)] is not None:
                need_test[f - 1].append((i, kind))
                need_test[f].append((i, kind))
    pts_cache = {}
    res = defaultdict(dict)  # i -> {(kind, fr): (n_in, n_vis)}
    n_rays = 0
    for fr in sorted(need_test):
        scene.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        cam_info = camera_frustum(scene, cam)
        for i, kind in need_test[fr]:
            empty = changes[i][1]
            if (i, kind) not in pts_cache:
                pts_cache[(i, kind)] = box_points(*boxes[(i, kind)])
            own = {empty.name} | {m.name for m in units[empty]}
            n_in, n_vis = count_visible(scene, dg, cam_info, pts_cache[(i, kind)], own)
            n_rays += n_in
            res[i][(kind, fr)] = (n_in, n_vis)
    t2 = time.time()

    details = [
        f"{len(changes)} changes checked; boxes {t1 - t0:.1f}s, rays {t2 - t1:.1f}s ({n_rays} rays)"
    ]
    n_fail = n_marg = 0
    for i, (f, empty) in enumerate(changes):
        parts, worst = [], 0
        for kind in ("old", "new"):
            if boxes[(i, kind)] is None:
                parts.append(f"{kind}=absent")
                continue
            for fr in (f - 1, f):
                n_in, n_vis = res[i][(kind, fr)]
                worst = max(worst, n_vis)
                parts.append(f"{kind}@{fr}={n_vis}/{n_in}")
        tag = "ok"
        if worst > 0:
            n_fail += 1
            if worst <= MARGINAL_POINTS:
                n_marg += 1
                tag = "FAIL(marginal)"
            else:
                tag = "FAIL(gross)"
        details.append(
            f"{tag:15s} {empty.name} f={f} [{unit_room(empty)}] " + " ".join(parts)
        )
    details.insert(
        1,
        f"visible changes: {n_fail} ({n_marg} marginal <= {MARGINAL_POINTS} pts, {n_fail - n_marg} gross)",
    )
    return details, n_fail == 0


# ---------------------------------------------------------------- check 3


def check_variation(units, frames_by_unit):
    multi = absent_back = 0
    examples = []
    for empty, members in units.items():
        states = [unit_state(empty, members, f) for f in frames_by_unit[empty]]
        if len(set(states)) >= 2:
            multi += 1
        absent = [all(s[2]) if s[2] else False for s in states]
        if any(a and not b for i, a in enumerate(absent) for b in absent[i + 1 :]):
            absent_back += 1
            if len(examples) < 5:
                examples.append(empty.name)
    details = [
        f"units with >=2 distinct states: {multi}/{len(units)}",
        f"units going absent -> present later: {absent_back}  e.g. {examples}",
    ]
    return details, multi >= 1 and absent_back >= 1


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blend")
    ap.add_argument("--max_changes", type=int, default=200)
    args = ap.parse_args()

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    scene = bpy.context.scene
    rooms = unhide_rooms(scene)
    cam = bpy.data.objects.get("Camera") or scene.camera
    scene.camera = cam
    print(f"blend: {args.blend}")
    print(
        f"frames {scene.frame_start}..{scene.frame_end}, {len(rooms)} room collections, camera={cam and cam.name}"
    )

    units = {
        o: list(o.children_recursive)
        for o in sorted(bpy.data.objects, key=lambda o: o.name)
        if o.name.startswith(UNIT_PREFIX) and o.type == "EMPTY"
    }

    # change frames: keyframes other than the first, where the state actually changes
    frames_by_unit = {}
    changes, noop = [], 0
    for empty, members in units.items():
        kf = key_frames(empty, members)
        frames_by_unit[empty] = kf
        for f in kf[1:]:
            if unit_state(empty, members, f) != unit_state(empty, members, f - 1):
                changes.append((f, empty))
            else:
                noop += 1
    changes.sort(key=lambda c: (c[0], c[1].name))
    total_changes = len(changes)
    all_changes = list(changes)
    if len(changes) > args.max_changes:
        idx = np.unique(
            np.linspace(0, len(changes) - 1, args.max_changes).round().astype(int)
        )
        changes = [changes[i] for i in idx]

    report("1. structure", *check_structure(scene, units))

    if cam is None:
        report("2. unseen-at-change", ["no camera"], False)
    elif not changes:
        report("2. unseen-at-change", ["no state changes to check (vacuous)"], True)
    else:
        report("2. unseen-at-change", *check_unseen(scene, cam, units, changes))

    report("3. variation", *check_variation(units, frames_by_unit))

    per_room = defaultdict(lambda: [0, 0])
    for empty in units:
        per_room[unit_room(empty)][0] += 1
    for _f, empty in all_changes:
        per_room[unit_room(empty)][1] += 1
    summary = [
        f"units: {len(units)}",
        f"state changes: {total_changes} (checked {len(changes)}, no-op keyframes {noop})",
    ] + [f"{room}: {u} units, {c} changes" for room, (u, c) in sorted(per_room.items())]
    print("\n[INFO] 4. summary")
    for line in summary:
        print(f"    {line}")

    failed = [n for n, ok in results if not ok]
    print(f"\nRESULT: {'FAIL ' + str(failed) if failed else 'ALL PASS'}")
    sys.stdout.flush()
    os._exit(1 if failed else 0)  # skip bpy teardown


if __name__ == "__main__":
    main()

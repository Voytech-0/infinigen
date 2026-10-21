"""Put furnishings in "superposition": each one may be absent or sit at one of several
candidate spots, and its state is only ever re-drawn while the camera cannot see it.

Runs once at scene creation, on a built house tour scene (or a loaded one being
re-saved). The state schedule is baked as constant-interpolation keyframes on one
Empty per unit, plus hide_render/hide_viewport keyframes on its members; render runs
just play them back.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree

logger = logging.getLogger(__name__)

_SHELL = (
    "house_floor",
    "house_ceiling",
    "house_wall_back",
    "house_wall_interior",
    "house_wall_exterior",
    "house_wall_corner",
    "house_wall_window",
    "house_wall_sill",
)
_WALLS = tuple(s for s in _SHELL if s.startswith("house_wall"))
_FLOOR_TOL = 0.05  # m, bottom this close to the floor = floor-standing
_SUPPORT_TOL = 0.04  # m, resting-on-top contact tolerance
_THIN = 0.05  # m, hosts thinner than this (rugs) don't carry what stands on them
_SHRINK = 0.01  # m, AABB overlap tolerance (touching is not colliding)
_BLOCKER_MARGIN = 0.1  # m, free space kept around static objects at new spots
_PATH_CLEARANCE = 0.6  # m, spots must stay this far (in plan) from the camera path
_SAMPLE_SPACING = 0.25  # m, visibility sample grid spacing on a state's box
_FRUSTUM_MARGIN = 1.1  # widen the frustum a little, to be safe at the image border
_NEAR = 0.05  # m
_OCCLUSION_TOL = 0.1  # m, a hit this close to the target does not occlude it
_FLOOR_TRIES = 300


@dataclass
class _Unit:
    roots: list[bpy.types.Object]
    lo: np.ndarray
    hi: np.ndarray
    room: bpy.types.Collection
    floor_standing: bool
    boxes_all: list = field(default_factory=list)  # (lo, hi) per root
    states: list = field(default_factory=list)  # (center, quarter_turns) or None
    boxes: list = field(default_factory=list)  # (lo, hi) per state, None if absent
    vis: np.ndarray | None = None  # [state, frame] seen by the camera
    current: int = 0
    pending: bool = False
    last_seen: int = -(10**9)  # frame index the current state was last seen
    changes: int = 0

    @property
    def center(self):
        return (self.lo + self.hi) / 2

    @property
    def volume(self):
        return float(np.prod(self.hi - self.lo))

    @property
    def members(self):
        return list(
            dict.fromkeys(o for r in self.roots for o in (r, *r.children_recursive))
        )


@contextmanager
def _rooms_unhidden(rooms):
    """Room collections are hidden in the viewport, which keeps their objects out of
    the evaluated depsgraph (no modifiers). Unhide them while measuring."""
    saved = [(c, c.hide_viewport) for c in rooms]
    for c in rooms:
        c.hide_viewport = False
    bpy.context.view_layer.update()
    try:
        yield bpy.context.evaluated_depsgraph_get()
    finally:
        for c, hidden in saved:
            c.hide_viewport = hidden


def _subtree_aabb(obj, depsgraph):
    corners = []
    for o in (obj, *obj.children_recursive):
        if o.type == "MESH":
            ev = o.evaluated_get(depsgraph)
            corners += [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    if not corners:
        return None
    corners = np.array(corners)
    return corners.min(axis=0), corners.max(axis=0)


def _overlap(lo_a, hi_a, lo_b, hi_b):
    return bool(np.all(lo_a + _SHRINK < hi_b) and np.all(lo_b + _SHRINK < hi_a))


def _state_box(unit, state):
    center, turns = state
    half = (unit.hi - unit.lo) / 2
    if turns % 2:
        half = half[[1, 0, 2]]
    return center - half, center + half


def _surface_points(box):
    lo, hi = box
    center, half = (lo + hi) / 2, (hi - lo) / 2 * 0.95
    lo, hi = center - half, center + half
    n = np.maximum(2, np.ceil((hi - lo) / _SAMPLE_SPACING).astype(int) + 1)
    axes = [np.linspace(a, b, k) for a, b, k in zip(lo, hi, n)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return grid[np.any((grid == lo) | (grid == hi), axis=1)]


def _transparent_tree(tree):
    """Any glass, transmission or alpha, recursing into groups. Linked inputs count
    as transparent (conservative: such objects stop occluding)."""
    for node in tree.nodes:
        if node.type in ("BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION"):
            return True
        if (
            node.type == "GROUP"
            and node.node_tree
            and _transparent_tree(node.node_tree)
        ):
            return True
        if node.type == "BSDF_PRINCIPLED":
            t, a = node.inputs["Transmission Weight"], node.inputs["Alpha"]
            if t.is_linked or t.default_value > 0:
                return True
            if a.is_linked or a.default_value < 1:
                return True
    return False


def _transparent(obj):
    return any(
        s.material and s.material.use_nodes and _transparent_tree(s.material.node_tree)
        for s in obj.material_slots
    )


def _build_bvh(objs, depsgraph):
    verts, tris, offset = [], [], 0
    for obj in objs:
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        mesh.calc_loop_triangles()
        v = np.empty(len(mesh.vertices) * 3, np.float32)
        mesh.vertices.foreach_get("co", v)
        t = np.empty(len(mesh.loop_triangles) * 3, np.int32)
        mesh.loop_triangles.foreach_get("vertices", t)
        world = np.array(ev.matrix_world)
        v = v.reshape(-1, 3).astype(np.float64) @ world[:3, :3].T + world[:3, 3]
        verts.append(v)
        tris.append(t.reshape(-1, 3) + offset)
        offset += len(v)
        ev.to_mesh_clear()
    if not verts:
        return None
    return BVHTree.FromPolygons(
        np.concatenate(verts).tolist(), np.concatenate(tris).tolist()
    )


def _box_bvh(lo, hi):
    lo, hi = lo + _SHRINK, hi - _SHRINK
    verts = [
        (x, y, z)
        for x in (lo[0], hi[0])
        for y in (lo[1], hi[1])
        for z in (lo[2], hi[2])
    ]
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    return BVHTree.FromPolygons(verts, quads)


class _Camera:
    def __init__(self, cam, scene, frames):
        loc = np.array(cam.location, dtype=float)[None].repeat(len(frames), 0)
        rot = np.array(cam.rotation_euler, dtype=float)[None].repeat(len(frames), 0)
        if cam.animation_data and cam.animation_data.action:
            for fc in cam.animation_data.action.fcurves:
                target = {"location": loc, "rotation_euler": rot}.get(fc.data_path)
                if target is not None:
                    target[:, fc.array_index] = [fc.evaluate(f) for f in frames]
        mode = cam.rotation_mode
        self.pos = loc
        self.rot = np.array([np.array(Euler(r, mode).to_matrix()) for r in rot])
        frame = np.array([v[:] for v in cam.data.view_frame(scene=scene)])
        self.tan = np.abs(frame[:, :2] / frame[:, 2:3]).max(axis=0)

    def visibility(self, points, bvh):
        """Per frame: is any of the points in the frustum and unoccluded."""
        d = points[None] - self.pos[:, None]
        cs = np.einsum("fpi,fij->fpj", d, self.rot)  # camera looks down -Z
        inside = (cs[..., 2] < -_NEAR) & np.all(
            np.abs(cs[..., :2]) <= _FRUSTUM_MARGIN * self.tan * -cs[..., 2:3], axis=-1
        )
        vis = np.zeros(len(self.pos), bool)
        for f in np.flatnonzero(inside.any(axis=1)):
            if bvh is None:
                vis[f] = True
                continue
            origin = Vector(self.pos[f])
            for ray in d[f, inside[f]]:
                dist = float(np.linalg.norm(ray))
                hit = bvh.ray_cast(origin, Vector(ray / dist), dist)
                if hit[0] is None or hit[3] > dist - _OCCLUSION_TOL:
                    vis[f] = True
                    break
        return vis


def _find_units(rooms, depsgraph):
    """Group each room's loose objects into rigid units (a table + what sits on it)."""
    units, blockers, occluders, walls = [], {}, [], {}
    for room in rooms:
        blockers[room.name], walls[room.name] = [], []
        eligible, lights = [], []
        for obj in room.objects:
            if obj.parent is not None:
                continue  # moves and hides with its root
            if obj.name.startswith("house_"):
                meshes = [o for o in (obj, *obj.children_recursive) if o.type == "MESH"]
                occluders += [o for o in meshes if not _transparent(o)]
                if obj.name.startswith(_WALLS):
                    walls[room.name] += meshes
                elif not obj.name.startswith(_SHELL) and meshes:
                    blockers[room.name].append(_subtree_aabb(obj, depsgraph))
            elif obj.type == "LIGHT":
                lights.append(obj)
            elif (box := _subtree_aabb(obj, depsgraph)) is not None:
                eligible.append((obj, *box))
        floors = [o for o in room.objects if o.name.startswith("house_floor")]
        floor_z = min((_subtree_aabb(f, depsgraph)[0][2] for f in floors), default=0.0)
        eligible.sort(key=lambda e: -np.prod(e[2] - e[1]))
        room_units = []
        for obj, lo, hi in eligible:
            c = (lo + hi) / 2
            host = None
            for u in room_units:
                for m_lo, m_hi in u.boxes_all:
                    if m_hi[2] - m_lo[2] < _THIN:
                        continue
                    inside = np.all(c > m_lo - 0.03) and np.all(c < m_hi + 0.03)
                    on_top = (
                        abs(lo[2] - m_hi[2]) < _SUPPORT_TOL
                        and np.all(c[:2] > m_lo[:2])
                        and np.all(c[:2] < m_hi[:2])
                    )
                    if inside or on_top:
                        host = u
                        break
                if host:
                    break
            if host:
                host.roots.append(obj)
                host.boxes_all.append((lo, hi))
                host.lo, host.hi = np.minimum(host.lo, lo), np.maximum(host.hi, hi)
            else:
                floor_standing = lo[2] < floor_z + _FLOOR_TOL
                u = _Unit([obj], lo.copy(), hi.copy(), room, floor_standing)
                u.boxes_all = [(lo, hi)]
                room_units.append(u)
        for light in lights:
            p = np.array(light.matrix_world.translation)
            for u in room_units:
                if np.all(p > u.lo - 0.03) and np.all(p < u.hi + 0.03):
                    u.roots.append(light)
                    break
        units += room_units
    return units, blockers, occluders, walls


def _floor_states(rng, u, spots, depsgraph, blockers, walls_bvh, path):
    floors = [o for o in u.room.objects if o.name.startswith("house_floor")]
    floor_bvh = _build_bvh(floors, depsgraph)
    if floor_bvh is None:
        return []
    boxes = [_subtree_aabb(f, depsgraph) for f in floors]
    f_lo = np.min([b[0] for b in boxes], axis=0)
    f_hi = np.max([b[1] for b in boxes], axis=0)

    def on_floor(x, y):
        hit = floor_bvh.ray_cast(Vector((x, y, f_hi[2] + 2)), Vector((0, 0, -1)))
        return hit[0] is not None

    states = []
    for _ in range(_FLOOR_TRIES):
        if len(states) >= spots:
            break
        c = np.array([*rng.uniform(f_lo[:2], f_hi[:2]), u.center[2]])
        state = (c, int(rng.integers(4)))
        lo, hi = _state_box(u, state)
        if not all(on_floor(x, y) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])):
            continue
        m = _BLOCKER_MARGIN
        if any(_overlap(lo - m, hi + m, b_lo, b_hi) for b_lo, b_hi in blockers):
            continue
        if walls_bvh is not None and walls_bvh.overlap(_box_bvh(lo, hi)):
            continue
        gap = np.maximum(np.maximum(lo[:2] - path, path - hi[:2]), 0)
        if (np.linalg.norm(gap, axis=1) < _PATH_CLEARANCE).any():
            continue
        states.append(state)
    return states


def make_superposition(
    rng: np.random.Generator,
    fraction: float = 0.5,
    spots: int = 3,
    camera: bpy.types.Object | None = None,
    cooldown: int = 100,
) -> dict:
    scene = bpy.context.scene
    camera = camera or scene.camera
    frames = np.arange(scene.frame_start, scene.frame_end + 1)
    cam = _Camera(camera, scene, frames)

    allowed = {int(i) for i in camera.get("allowed_rooms", [])}
    rooms = [
        c
        for c in bpy.data.collections
        if c.name.startswith("house_room_") and int(c.name.rsplit("_", 1)[1]) in allowed
    ]
    with _rooms_unhidden(rooms) as depsgraph:
        units, blockers, occluders, walls = _find_units(rooms, depsgraph)
        if not units:
            logger.warning("Superposition: no units found in rooms %s", sorted(allowed))
            return {"units": 0, "quantum": 0, "changes": 0}
        n_quantum = max(1, int(round(fraction * len(units))))
        quantum_ids = set(
            rng.choice(len(units), size=n_quantum, replace=False).tolist()
        )
        quantum = [u for i, u in enumerate(units) if i in quantum_ids]
        static = [u for i, u in enumerate(units) if i not in quantum_ids]
        logger.info(
            "Superposition: %d units (%d quantum, %d static)",
            len(units),
            len(quantum),
            len(static),
        )
        # static big units occlude; small ones are ignored (conservative: occludes less)
        for u in static:
            blockers[u.room.name] += u.boxes_all
            if u.floor_standing or u.volume > 0.05:
                occluders += [
                    m for m in u.members if m.type == "MESH" and not _transparent(m)
                ]
        bvh = _build_bvh(occluders, depsgraph)
        walls_bvh = {name: _build_bvh(objs, depsgraph) for name, objs in walls.items()}

        for u in quantum:
            u.states = [(u.center, 0)]
            if u.floor_standing:
                u.states += _floor_states(
                    rng,
                    u,
                    spots,
                    depsgraph,
                    blockers[u.room.name],
                    walls_bvh[u.room.name],
                    cam.pos[:, :2],
                )
            u.states.append(None)  # absent
            u.boxes = [None if s is None else _state_box(u, s) for s in u.states]
            vis = [cam.visibility(_surface_points(b), bvh) for b in u.boxes[:-1]]
            # absent is seen whenever any spot it could occupy is seen
            u.vis = np.array([*vis, np.any(vis, axis=0)])

    log = _schedule(rng, quantum, frames, cooldown)
    _bake(quantum, log)
    stats = {
        "units": len(units),
        "quantum": len(quantum),
        "states": sum(len(u.states) for u in quantum),
        "changes": sum(u.changes for u in quantum),
        "absent_at_start": sum(
            u.states[log[n][0][1]] is None for n, u in enumerate(quantum)
        ),
        "reappearances": sum(
            u.states[a] is None and u.states[b] is not None
            for n, u in enumerate(quantum)
            for (_, a), (_, b) in zip(log[n], log[n][1:])
        ),
    }
    logger.info("Superposition baked: %s", stats)
    return stats


def _schedule(rng, quantum, frames, cooldown):
    """Per unit, the (frame, state) list of its state changes.

    A unit only changes after its current state has been unseen for `cooldown` frames.
    """

    def conflicts(u, box):
        return any(
            v is not u
            and v.boxes[v.current] is not None
            and _overlap(*box, *v.boxes[v.current])
            for v in quantum
        )

    log = [None] * len(quantum)
    for n in rng.permutation(len(quantum)):
        u = quantum[n]
        u.current = len(u.states) - 1  # absent unless something fits
        for k in rng.permutation(len(u.states)):
            if u.boxes[k] is None or not conflicts(u, u.boxes[k]):
                u.current = int(k)
                break
        log[n] = [(int(frames[0]), u.current)]

    for i in range(1, len(frames)):
        for n, u in enumerate(quantum):
            seen = u.vis[:, i] | u.vis[:, i - 1]
            if seen[u.current]:
                u.pending = True
                u.last_seen = i
                continue
            if not u.pending or i - u.last_seen < cooldown:
                continue
            for k in rng.permutation(len(u.states)):
                k = int(k)
                if k != u.current:
                    if seen[k] or (u.boxes[k] is not None and conflicts(u, u.boxes[k])):
                        continue
                    log[n].append((int(frames[i]), k))
                    u.current = k
                    u.changes += 1
                u.pending = False
                break
    return log


def _bake(quantum, log):
    for n, u in enumerate(quantum):
        empty = bpy.data.objects.new(f"quantum_unit_{n:03d}", None)
        u.room.objects.link(empty)
        for r in u.roots:
            r.parent = empty
            r.matrix_parent_inverse = Matrix.Identity(4)
        members = u.members
        for f, k in log[n]:
            state = u.states[k]
            absent = state is None
            if not absent:
                center, turns = state
                rot = Matrix.Rotation(np.pi / 2 * turns, 3, "Z")
                empty.rotation_euler = rot.to_euler()
                empty.location = Vector(center) - rot @ Vector(u.center)
                empty.keyframe_insert("location", frame=f)
                empty.keyframe_insert("rotation_euler", frame=f)
            for m in members:
                m.hide_render = absent
                m.hide_viewport = absent
                m.keyframe_insert("hide_render", frame=f)
                m.keyframe_insert("hide_viewport", frame=f)
        for o in [empty, *members]:
            if o.animation_data and o.animation_data.action:
                for fc in o.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"

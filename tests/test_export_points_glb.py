"""Demo-style scene.glb (points + camera wireframes) and our own PLY reader layout."""

import json
import struct

import numpy as np

from da3.export import camera_frustum_segments, write_ply, write_points_glb


def _glb_json(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n])


def test_frustum_segments_start_at_the_camera_centre_and_reach_the_corners():
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = np.linalg.inv(c2w)[None]
    segs = camera_frustum_segments(w2c, K[None], (80, 100), size=2.0)
    assert segs.shape == (8, 2, 3)
    assert np.allclose(segs[:4, 0], [1.0, 2.0, 3.0])                         # apex = camera centre
    assert np.allclose(segs[:4, 1, 2], 3.0 + 2.0)                            # corners at depth `size`
    assert np.allclose(segs[0, 1], [1.0 + (0 - 50) / 100 * 2, 2.0 + (0 - 40) / 100 * 2, 5.0])


def test_points_glb_has_a_points_primitive_and_camera_lines(tmp_path):
    rng = np.random.default_rng(0)
    P = rng.uniform(-1, 1, (500, 3))
    C = rng.integers(0, 255, (500, 3), dtype=np.uint8)
    w2c = np.tile(np.eye(4)[:3], (3, 1, 1))
    w2c[:, 2, 3] = [2.0, 3.0, 4.0]
    K = np.tile(np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]]), (3, 1, 1))
    n = write_points_glb(tmp_path / "scene.glb", P, C, w2c, K, (80, 100))
    js = _glb_json(tmp_path / "scene.glb")
    modes = [p.get("mode", 4) for m in js["meshes"] for p in m["primitives"]]
    assert n == 500 and 0 in modes and 1 in modes                            # POINTS + LINES, like the demo
    pts = next(p for m in js["meshes"] for p in m["primitives"] if p.get("mode") == 0)
    assert js["accessors"][pts["attributes"]["POSITION"]]["count"] == 500 and "COLOR_0" in pts["attributes"]
    write_points_glb(tmp_path / "nocam.glb", P, C)                           # points only
    assert [p.get("mode") for m in _glb_json(tmp_path / "nocam.glb")["meshes"] for p in m["primitives"]] == [0]


def test_ply_layout_matches_the_blender_reader(tmp_path):
    """scripts/blender_import.py reads our PLY with numpy: it relies on exactly this header and record layout."""
    P = np.arange(12, dtype=np.float32).reshape(4, 3)
    C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], np.uint8)
    write_ply(tmp_path / "p.ply", P, C)
    raw = (tmp_path / "p.ply").read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    lines = head.decode().strip().splitlines()
    assert "format binary_little_endian 1.0" in lines and "element vertex 4" in lines
    assert [l.split()[-1] for l in lines if l.startswith("property")] == ["x", "y", "z", "red", "green", "blue"]
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.frombuffer(body, dtype=dt, count=4)
    assert np.allclose(np.stack([arr["x"], arr["y"], arr["z"]], 1), P) and (arr["blue"] == C[:, 2]).all()

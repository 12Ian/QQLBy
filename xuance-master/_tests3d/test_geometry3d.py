import numpy as np
from xuance.environment.multi_agent_env import geometry3d as g


def test_fibonacci_sphere_unit_and_count():
    dirs = g.fibonacci_sphere(200)
    assert dirs.shape == (200, 3)
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_solid_angle_sum_is_4pi():
    n = 200
    total = n * g.solid_angle_per_point(n)
    assert abs(total - 4 * np.pi) < 1e-9


def test_fibonacci_roughly_uniform():
    # mean direction of a uniform sphere sampling is ~0
    dirs = g.fibonacci_sphere(2000)
    assert np.linalg.norm(dirs.mean(axis=0)) < 0.05


def test_ray_free_distance_world_box_no_buildings():
    # origin at center, +x ray hits x=1000 wall at distance 500
    origin = np.array([500., 500., 200.], np.float32)
    dirs = np.array([[1., 0., 0.], [0., 0., 1.]], np.float32)
    d = g.ray_free_distance(origin, dirs,
                            map_size=1000., z_min=50., z_max=350.,
                            buildings=np.zeros((0, 4), np.float32))
    assert abs(d[0] - 500.) < 1e-3          # +x to wall
    assert abs(d[1] - 150.) < 1e-3          # +z to ceiling (350-200)


def test_ray_free_distance_blocked_by_building():
    origin = np.array([100., 500., 200.], np.float32)
    dirs = np.array([[1., 0., 0.]], np.float32)
    buildings = np.array([[300., 350., 480., 520.]], np.float32)  # ahead on +x
    d = g.ray_free_distance(origin, dirs, 1000., 50., 350., buildings)
    assert abs(d[0] - 200.) < 1.0           # blocked at x=300 -> dist 200


def test_footprint_los_clear_and_blocked():
    buildings = np.array([[400., 600., 400., 600.]], np.float32)
    p1 = np.array([100., 500., 100.], np.float32)
    p2 = np.array([900., 500., 300.], np.float32)   # passes through block in xy
    assert g.footprint_los(p1, p2, buildings) is False
    p3 = np.array([100., 100., 300.], np.float32)    # clears block in xy
    assert g.footprint_los(p1, p3, buildings) is True


def test_footprint_los_no_buildings_true():
    p1 = np.array([0., 0., 0.], np.float32)
    p2 = np.array([1000., 1000., 300.], np.float32)
    assert g.footprint_los(p1, p2, np.zeros((0, 4), np.float32)) is True


def test_radar_dirs_shape_and_unit():
    dirs = g.radar_body_dirs()
    assert dirs.shape == (16, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-5)


def test_radar_distances_open_space_is_one():
    origin = np.array([500., 500., 200.], np.float32)
    out = g.radar_distances(origin, yaw=0.0, pitch=0.0, radar_range=100.0,
                            map_size=1000., z_min=50., z_max=350.,
                            buildings=np.zeros((0, 4), np.float32))
    assert out.shape == (16,)
    assert np.all(out >= 0.99)   # nothing within 100m -> clipped to 1.0


def test_footprint_los_batch_matches_scalar():
    rng = np.random.default_rng(0)
    buildings = np.array([[400., 600., 400., 600.],
                          [200., 250., 700., 760.]], np.float32)
    origin = np.array([100., 500., 120.], np.float32)
    pts = rng.uniform(0, 1000, size=(40, 3)).astype(np.float32)
    pts[:, 2] = 200.
    batch = g.footprint_los_batch(origin, pts, buildings)
    assert batch.shape == (40,)
    scalar = np.array([g.footprint_los(origin, p, buildings) for p in pts])
    assert np.array_equal(batch, scalar)


def test_footprint_los_batch_no_buildings_all_true():
    origin = np.array([0., 0., 0.], np.float32)
    pts = np.array([[1000., 1000., 300.], [500., 10., 100.]], np.float32)
    out = g.footprint_los_batch(origin, pts, np.zeros((0, 4), np.float32))
    assert out.dtype == bool and out.shape == (2,) and out.all()

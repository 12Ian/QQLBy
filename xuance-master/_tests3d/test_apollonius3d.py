import numpy as np
from xuance.environment.multi_agent_env import geometry3d as g
from xuance.environment.multi_agent_env import apollonius3d as a


def _setup(n=200):
    dirs = g.fibonacci_sphere(n)
    target = np.array([500., 500., 200.], np.float32)
    free = g.ray_free_distance(target, dirs, 1000., 50., 350.,
                               np.zeros((0, 4), np.float32))
    return dirs, target, free


def test_open_field_large_escape_angle():
    # one slow far pursuer -> most directions dangerous -> big escape solid angle
    dirs, target, free = _setup()
    uav = np.array([[100., 100., 200.]], np.float32)
    out = a.compute_escape_field_3d(target, target_speed=10.0,
            uav_positions=uav, uav_speed_max=10.5, dirs=dirs, free_dist=free,
            catch_radius=15.0, r_f=120.0, margin_ema=None, dangerous_prev=None)
    assert out["escape_solid_angle"] > 0.5 * 4 * np.pi


def test_surrounded_small_escape_angle():
    # 12 fast close pursuers on a shell -> few/no dangerous dirs -> Omega -> 0
    dirs, target, free = _setup()
    shell = g.fibonacci_sphere(12) * 30.0 + target
    out = a.compute_escape_field_3d(target, target_speed=4.0,
            uav_positions=shell.astype(np.float32), uav_speed_max=11.0,
            dirs=dirs, free_dist=free, catch_radius=15.0, r_f=60.0,
            margin_ema=None, dangerous_prev=None)
    assert out["escape_solid_angle"] < 0.2 * 4 * np.pi


def test_keys_and_shapes():
    dirs, target, free = _setup()
    uav = np.array([[100., 100., 200.], [900., 900., 200.]], np.float32)
    out = a.compute_escape_field_3d(target, 8.0, uav, 10.5, dirs, free,
                                    15.0, 120.0, None, None)
    for k in ["geometry_open", "dangerous", "controlled", "best_owner",
              "control_strength", "escape_solid_angle", "margin_ema",
              "dangerous_prev"]:
        assert k in out
    assert out["dangerous"].shape == (dirs.shape[0],)
    assert out["best_owner"].shape == (dirs.shape[0],)


def test_visibility_equals_euclidean_without_buildings():
    dirs, target, free = _setup()
    uav = np.array([[540., 500., 200.], [500., 560., 200.]], np.float32)
    common = dict(target_pos=target, target_speed=6.0, uav_positions=uav,
                  uav_speed_max=11.0, dirs=dirs, free_dist=free,
                  catch_radius=15.0, r_f=120.0, margin_ema=None, dangerous_prev=None)
    euc = a.compute_escape_field_3d(**common, criterion_mode="euclidean", buildings=None)
    vis = a.compute_escape_field_3d(**common, criterion_mode="visibility",
                                    buildings=np.zeros((0, 4), np.float32))
    assert np.array_equal(euc["dangerous"], vis["dangerous"])
    assert abs(euc["escape_solid_angle"] - vis["escape_solid_angle"]) < 1e-9


def test_visibility_monotonic_and_strict_with_occluder():
    # one nearby pursuer controls a chunk in euclidean; a building between it and
    # the probe shell occludes it -> those dirs stay dangerous -> Omega grows.
    dirs = g.fibonacci_sphere(200)
    target = np.array([500., 500., 200.], np.float32)
    buildings = np.array([[505., 535., 505., 525.]], np.float32)  # NE of target
    free = g.ray_free_distance(target, dirs, 1000., 50., 350., buildings)
    uav = np.array([[500., 560., 200.]], np.float32)  # 60m north, close & fast
    common = dict(target_pos=target, target_speed=4.0, uav_positions=uav,
                  uav_speed_max=11.0, dirs=dirs, free_dist=free,
                  catch_radius=15.0, r_f=120.0, margin_ema=None, dangerous_prev=None)
    euc = a.compute_escape_field_3d(**common, criterion_mode="euclidean", buildings=buildings)
    vis = a.compute_escape_field_3d(**common, criterion_mode="visibility", buildings=buildings)
    assert vis["escape_solid_angle"] >= euc["escape_solid_angle"] - 1e-9   # monotonic
    assert vis["escape_solid_angle"] > euc["escape_solid_angle"] + 1e-6    # strict
    assert vis["criterion_mode"] == "visibility"

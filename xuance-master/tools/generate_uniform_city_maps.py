"""Reproduce the five fixed uniform city maps without a random generator."""
import argparse
import json
from pathlib import Path


COUNTS = (20, 40, 60, 80, 100)
SPECS = {
    1: {"sides": (20, 40), "heights": (10, 60), "per_level": 10},
    2: {"sides": (40, 60), "heights": (60, 150), "per_level": 8},
    3: {"sides": (60, 80), "heights": (150, 350), "per_level": 2},
}


def build_city():
    buildings = []
    for rank in range(5):
        centers = [(row, col) for row in range(10) for col in range(10)
                   if (row + 2 * col) % 5 == rank]
        type_indices = {1: 0, 2: 0, 3: 0}
        for index, (row, col) in enumerate(centers):
            ordinary_slot = (row + rank) % 2
            landmark_rows = {(2 * rank + 2) % 10, (2 * rank + 7) % 10}
            kind = 1 if index % 2 == ordinary_slot else (3 if row in landmark_rows else 2)
            spec = SPECS[kind]
            local_index = type_indices[kind]
            type_indices[kind] += 1
            small, large = spec["sides"]
            sizes = [(small, small), (small, large), (large, small), (large, large)]
            width, length = sizes[(local_index + rank) % 4]
            low, high = spec["heights"]
            height_index = (3 * local_index + rank) % spec["per_level"]
            height = round((low + (high - low) * height_index / (spec["per_level"] - 1)) / 5) * 5
            center_x, center_y = 50 + 100 * col, 50 + 100 * row
            buildings.append({
                "id": len(buildings) + 1, "added_at_level": rank + 1,
                "type": kind, "row": row, "column": col,
                "width": width, "length": length, "top": height,
                "xmin": center_x - width / 2, "xmax": center_x + width / 2,
                "ymin": center_y - length / 2, "ymax": center_y + length / 2,
            })
    return buildings


def validate(buildings):
    for level, count in enumerate(COUNTS, start=1):
        subset = buildings[:count]
        quotas = [sum(b["type"] == kind for b in subset) for kind in SPECS]
        if len(subset) != count or quotas != [count // 2, count * 2 // 5, count // 10]:
            raise ValueError("Incorrect building class quotas")
        for axis in ("row", "column"):
            if any(sum(b[axis] == i for b in subset) != 2 * level for i in range(10)):
                raise ValueError("Uneven distribution across rows or columns")
    for b in buildings:
        spec = SPECS[b["type"]]
        if not (0 <= b["xmin"] < b["xmax"] <= 1000 and
                0 <= b["ymin"] < b["ymax"] <= 1000 and
                spec["heights"][0] <= b["top"] <= spec["heights"][1] and
                b["width"] in spec["sides"] and b["length"] in spec["sides"]):
            raise ValueError("Invalid building dimensions")
    for i, a in enumerate(buildings):
        for b in buildings[i + 1:]:
            dx = max(a["xmin"] - b["xmax"], b["xmin"] - a["xmax"], 0)
            dy = max(a["ymin"] - b["ymax"], b["ymin"] - a["ymax"], 0)
            if (dx * dx + dy * dy) ** .5 < 20 - 1e-6:
                raise ValueError("Building clearance below 20 m")


def save_maps(directory):
    buildings = build_city()
    validate(buildings)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layout": "deterministic_uniform", "random_generation": False,
        "map_size_xy_m": [1000, 1000], "candidate_center_spacing_m": 100,
        "minimum_footprint_clearance_m": 20, "type_ratios": [.5, .4, .1],
        "source": "tools/generate_uniform_city_maps.py",
        "building_details": buildings, "stages": [],
    }
    for level, count in enumerate(COUNTS, start=1):
        subset = buildings[:count]
        filename = f"buildings_uniform_{count}.json"
        rows = [[b[k] for k in ("xmin", "xmax", "ymin", "ymax", "top")] for b in subset]
        (directory / filename).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        manifest["stages"].append({
            "level": level, "file": filename, "count": count,
            "footprint_percent": round(sum(b["width"] * b["length"] for b in subset) / 10000, 2),
            "type_counts": [sum(b["type"] == kind for b in subset) for kind in SPECS],
            "tops_at_least_80m": sum(b["top"] >= 80 for b in subset),
            "tops_below_50m": sum(b["top"] < 50 for b in subset),
        })
    (directory / "uniform_map_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "xuance/environment/multi_agent_env/buildings")
    save_maps(parser.parse_args().output)

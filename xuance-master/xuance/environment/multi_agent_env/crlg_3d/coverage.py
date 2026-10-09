"""Monte Carlo union coverage of the predicted target endpoint cloud."""

import numpy as np


def _inside_polygon(samples, vertices):
    inside = np.zeros(len(samples), dtype=bool)
    if np.max(np.linalg.norm(vertices - vertices[0], axis=1)) < 1e-9:
        return inside
    x, y = samples[:, 0], samples[:, 1]
    previous = vertices[-1]
    for current in vertices:
        crossing = ((current[1] > y) != (previous[1] > y))
        x_cross = (previous[0] - current[0]) * (y - current[1]) / (
            previous[1] - current[1] + 1e-12) + current[0]
        inside ^= crossing & (x < x_cross)
        previous = current
    return inside


def coverage_probability(mean, covariance, centers, radii, normal_samples,
                         polygons=None, plane=None, zero_endpoints=None,
                         bounds=None):
    if len(centers) == 0:
        return 0.
    covariance = (covariance + covariance.T) / 2 + np.eye(2) * 1e-6
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-8)))
    samples = np.asarray(mean) + normal_samples @ factor.T
    covered = np.zeros(len(samples), dtype=bool)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        if polygons is not None and polygons[i] is not None:
            reached = _inside_polygon(samples, polygons[i])
        else:
            reached = np.sum((samples - center) ** 2, axis=1) <= radius ** 2
        if bounds is not None:
            reference = zero_endpoints[i]
            world_points = reference + (samples - center) @ plane
            reached &= np.all((world_points > bounds[0]) &
                              (world_points < bounds[1]), axis=1)
        covered |= reached
    return float(np.mean(covered))

import numpy as np
from numpy.random import normal, uniform, poisson

import utils

np.random.seed(1001)


scripter = utils.Scripter()


def circle(x=0, y=0, r=1, n_points=100, angle=0.0):
    """Generates a circle.

    Args:
        x (int, optional): x coordinate of the center of the circle. Defaults to 0.
        y (int, optional): y coordinate of the center of the circle. Defaults to 0.
        r (int, optional): Radius of the circle. Defaults to 1.
        n_points (int, optional): Number of points in the circle. Defaults to 100.
        angle (float, optional): Rotation of the circle. Only needed to prevent the first point of the circle to be always positioned at (x+r,y). Defaults to 0.0.

    Returns:
        np.ndarray: (n_points x 2) np.array consisting of the x and y coordinates of the circle.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points + 1)[:-1] + angle
    x = x + np.cos(theta) * r
    y = y + np.sin(theta) * r
    return x, y


def polygon(x=0, y=0, r=1, n_points=100, angle=0.0, n_sides=3):
    """
    Generates a polygon with n_sides sides, with the geometrical center
    at (x,y), and fitting inside a circle of radius r.

    The number of points per side will be `n_points // n_sides`, so the total
    number of points may not be exactly the specified `n_points` argument.
    """
    n_per_side = n_points // n_sides

    # Divide the polygon into n_sides triangles, e.g. a hexagon would have 6 triangles:
    #    _______
    #   /\     /\
    #  /  \   /  \
    # /    \ /    \
    # _____________
    # \    / \    /
    #  \  /   \  /
    #   \/_____\/

    # Define the angle at the center as 2*alpha,
    # the height as h, and the width as 2*w:

    #    /\ <--- 2*alpha   ^    ^
    #   /  \               |     \
    #  /    \              | h    \  r
    # /______\             v       v
    #
    # <------>
    #   2*w

    # I.e.:
    # |\  <--- alpha      ^     ^
    # | \                 |      \
    # |  \                | h     \  r
    # |___\               v        v

    # <--->
    #   w

    alpha = np.pi / n_sides  # 2*pi / (2*n_sides) = pi / n_sides
    l = np.cos(alpha) * r
    w = np.sin(alpha) * r

    points = []

    for i in range(n_sides):
        # Generate a straight line from x = -w to w, at y=1
        # This would be the upper side of the polygon
        p = np.row_stack(
            (
                np.linspace(-w, w, n_per_side, endpoint=False),  # x
                np.ones(n_per_side) * l,  # y
            )
        )
        # Rotate the straight line but 1, 3, ... * alpha
        rot = (2 * i + 1) * alpha
        rot_mat = np.array(
            [
                [np.cos(rot), -np.sin(rot)],
                [np.sin(rot), np.cos(rot)],
            ]
        )
        p = rot_mat.dot(p)
        points.append(p)

    points = np.hstack(points)
    assert points.shape[0] == 2

    # Rotate by the final angle
    points = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    ).dot(points)
    # Shift and return
    return points[0] + x, points[1] + y


def event(shuffle=True):
    """
    Generate an 'event': a 2D point cloud with noisee and a variable
    number of triangles, circles, and squares.

    Returns
    -------
    int

    """
    X = []
    y = []

    n_noise = int(normal(150, 20))
    X.append(uniform(-1, 1, (n_noise, 2)))
    y.append(np.zeros(n_noise, dtype=int))

    def gen_coords():
        angle = uniform(0, 2 * np.pi)
        x_c = uniform(-0.9, 0.9)
        y_c = uniform(-0.9, 0.9)
        r = uniform(0.1, 0.9)
        # Limit r so that shapes don't go beyond -1..1 on either axis
        extremity = max(abs(x_c - r), abs(x_c + r), abs(y_c - r), abs(y_c + r))
        if extremity > 1.0:
            r = min(abs(1 - x_c), abs(-1 - x_c), abs(1 - y_c), abs(-1 - y_c))
        n_points = int(60 * normal(1, 0.1) * r + 10)
        return x_c, y_c, angle, r, n_points

    shape_noise = 0.007

    # Add some triangles
    for _ in range(int(uniform(1, 3))):
        x_c, y_c, angle, r, n_points = gen_coords()
        x_shape, y_shape = polygon(
            x_c, y_c, r, angle=angle, n_sides=3, n_points=n_points
        )
        n_points = x_shape.shape[0]
        x_shape += normal(0.0, shape_noise, size=n_points)
        y_shape += normal(0.0, shape_noise, size=n_points)
        X.append(np.column_stack((x_shape, y_shape)))
        y.append(np.ones(n_points) * (y[-1][-1] + 1))

    # Add some squares
    for _ in range(int(uniform(0, 3))):
        x_c, y_c, angle, r, n_points = gen_coords()
        x_shape, y_shape = polygon(
            x_c, y_c, r, angle=angle, n_sides=4, n_points=n_points
        )
        n_points = x_shape.shape[0]
        x_shape += normal(0.0, shape_noise, size=n_points)
        y_shape += normal(0.0, shape_noise, size=n_points)
        X.append(np.column_stack((x_shape, y_shape)))
        y.append(np.ones(n_points) * (y[-1][-1] + 1))

    # Add some circles
    for _ in range(int(uniform(3, 1))):
        x_c, y_c, angle, r, n_points = gen_coords()
        x_shape, y_shape = circle(x_c, y_c, r, angle=angle, n_points=n_points)
        n_points = x_shape.shape[0]
        x_shape += normal(0.0, shape_noise, size=n_points)
        y_shape += normal(0.0, shape_noise, size=n_points)
        X.append(np.column_stack((x_shape, y_shape)))
        y.append(np.ones(n_points) * (y[-1][-1] + 1))

    X = np.vstack(X)
    y = np.concatenate(y)

    if shuffle:
        order = np.arange(X.shape[0])
        np.random.shuffle(order)
        X = X[order]
        y = y[order]

    return X, y


@scripter
def draw_circle():
    with utils.quick_ax() as ax:
        ax.scatter(*circle(x=2, y=5, r=3, n_points=50, angle=0.1 * np.pi))
        ax.scatter(*circle(x=1, y=1, r=2, n_points=40, angle=0.3 * np.pi))
        x, y = circle(x=6, y=6, r=2, n_points=40, angle=0.3 * np.pi)
        x += normal(0.0, 0.1, size=x.shape)
        y += normal(0.0, 0.1, size=x.shape)
        ax.scatter(x, y)
        ax.set_xlim(-2, 10)
        ax.set_ylim(-2, 10)


@scripter
def draw_polygon():
    with utils.quick_ax() as ax:
        x, y = polygon(n_points=60)
        ax.scatter(x[:20], y[:20], c="r")
        ax.scatter(x[20:40], y[20:40], c="g")
        ax.scatter(x[40:], y[40:], c="b")
        ax.scatter(*polygon(n_sides=4))
        ax.scatter(*polygon(n_sides=5))
        ax.scatter(*polygon(x=2.5, y=2.5, r=2, angle=0.1 * np.pi, n_sides=8))
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)


@scripter
def draw_event():
    n = utils.pull_arg("-n", type=int, default=5).n
    s = utils.pull_arg("-s", type=int, default=1001).s
    nocolor = utils.pull_arg("--nocolor", action="store_true").nocolor
    drawedges = utils.pull_arg("--drawedges", action="store_true").drawedges

    np.random.seed(s)
    for _ in range(n):
        with utils.quick_ax() as ax:
            X, y = event()
            print(f"n_shapes: {np.max(y)}; n_points: {X.shape[0]}")

            if drawedges:
                from sklearn.neighbors import NearestNeighbors

                nbrs = NearestNeighbors(n_neighbors=8, algorithm="ball_tree").fit(X)
                distances, indices = nbrs.kneighbors(X)

                for i in range(len(indices)):
                    for j in indices[i]:
                        ax.plot(
                            [X[i, 0], X[j, 0]],
                            [X[i, 1], X[j, 1]],
                            c="gray",
                            linewidth=0.4,
                        )

            if nocolor:
                ax.scatter(X[:, 0], X[:, 1])
            else:
                for i in np.unique(y):
                    sel = y == i
                    ax.scatter(X[sel, 0], X[sel, 1])

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)


if __name__ == "__main__":
    scripter.run()

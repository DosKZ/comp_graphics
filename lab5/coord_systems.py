import numpy as np


def normalize(vector):
    return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector


def rotation_xyz(angles):
    def rotation_x(angle):
        matrix = np.array([[1, 0, 0, 0],
                           [0, np.cos(angle), -np.sin(angle), 0],
                           [0, np.sin(angle), np.cos(angle), 0],
                           [0, 0, 0, 1]])
        return matrix

    def rotation_y(angle):
        matrix = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                           [0, 1, 0, 0],
                           [-np.sin(angle), 0, np.cos(angle), 0],
                           [0, 0, 0, 1]])
        return matrix

    def rotation_z(angle):
        matrix = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                           [np.sin(angle), np.cos(angle), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        return matrix

    x_matrix = rotation_x(angles[0])
    y_matrix = rotation_y(angles[1])
    z_matrix = rotation_z(angles[2])

    return x_matrix @ y_matrix @ z_matrix


def transfer_matrix(c):
    matrix = np.array([
        [1, 0, 0, c[0]],
        [0, 1, 0, c[1]],
        [0, 0, 1, c[2]],
        [0, 0, 0, 1]
    ])
    return matrix


def scale_matrix(s):
    matrix = np.array([
        [s[0], 0, 0, 0],
        [0, s[1], 0, 0],
        [0, 0, s[2], 0],
        [0, 0, 0, 1]
    ])
    return matrix


def mo2w(model_vertexes, angles, c, s, normal=False):
    R = rotation_xyz(angles)
    T = transfer_matrix(c)
    S = scale_matrix(s)
    M = R @ T @ S
    if normal:
        return np.dot(np.linalg.inv(M.T), model_vertexes.T).T
    else:
        return np.dot(M, model_vertexes.T).T


def mw2c(model_vertexes, cam_point, obj_point, normal=False):
    Tc = transfer_matrix(-cam_point)
    g = normalize(cam_point - obj_point)
    b = normalize(np.array([0, 1, 0]) - g[1] * g)
    a = normalize(np.cross(g, b))
    Rc = np.array([
        [a[0], b[0], g[0], 0],
        [a[1], b[1], g[1], 0],
        [a[2], b[2], g[2], 0],
        [0, 0, 0, 1]]
    )
    M = Rc @ Tc
    if normal:
        return np.dot(np.linalg.inv(M.T), model_vertexes.T).T
    else:
        return np.dot(M, model_vertexes.T).T


def orthogonal_projection(model_vertexes):
    x, y, z = model_vertexes[:, 0], model_vertexes[:, 1], model_vertexes[:, 2]
    l, r = np.min(x), np.max(x)
    b, t = np.min(y), np.max(y)
    n, f = np.min(z), np.max(z)

    ortho_matrix = np.array([[2 / (r - l), 0, 0, -((r + l) / (r - l))],
                             [0, 2 / (t - b), 0, -((t + b) / (t - b))],
                             [0, 0, 2 / (f - n), -((f + n) / (f - n))],
                             [0, 0, 0, 1]])

    return np.dot(ortho_matrix, model_vertexes.T).T


def viewport(model_vertexes, xy=(0,0), size=(1024,1024)):
    temp = model_vertexes.copy()
    ox = xy[0] + (size[0] / 2)
    oy = xy[1] + (size[1] / 2)
    temp[:, 0] = (size[0] / 2) * temp[:, 0] + ox
    temp[:, 1] = (size[1] / 2) * temp[:, 1] + oy

    return np.rint(temp).astype(int)[:, :-1]

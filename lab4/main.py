import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def q_gen(P):
    Q = []
    for i in range(P.shape[0]):
        if i == P.shape[0] - 1:
            Q.append(0.5 * (P[0] + P[i]))
        else:
            Q.append(0.5 * (P[i] + P[i + 1]))
    return np.array(Q)


def line(img, color, x0, y0, x1, y1):
    sign_x = 0
    sign_y = 0
    if x1 - x0 > 0:
        sign_x = 1
    elif x1 - x0 < 0:
        sign_x = -1
    if y1 - y0 > 0:
        sign_y = 1
    elif y1 - y0 < 0:
        sign_y = -1
    delta_x = abs(x1 - x0)
    delta_y = abs(y1 - y0)
    if delta_x > delta_y:
        d = delta_x
        dd = delta_y
    else:
        d = delta_y
        dd = delta_x
    x_cur, y_cur = x0, y0
    error = d / 2
    img[x_cur, y_cur] = color
    for i in range(d):
        error -= dd
        if error < 0:
            error += d
            x_cur += sign_x
            y_cur += sign_y
        else:
            if delta_x > delta_y:
                x_cur += sign_x
            else:
                y_cur += sign_y
        img[x_cur, y_cur] = color


def draw(img, color, P, t):
    Q = q_gen(P)
    for i in range(Q.shape[0]):
        points = []
        for j in t:
            if i == Q.shape[0] - 1:
                points.append((Q[0] * ((1 - j) ** 2) + 2 * P[0] * (1 - j) * j + Q[i] * (j ** 2)).astype(int))
            else:
                points.append((Q[i] * ((1 - j) ** 2) + 2 * P[i + 1] * (1 - j) * j + Q[i + 1] * (j ** 2)).astype(int))
        for j in range(len(points) - 1):
            line(img, color, points[j][0], points[j][1], points[j + 1][0],
                 points[j + 1][1])
    return plt.imshow(img)


def frames_creation(img, color, frames, center, r_list, rng, t, angles):
    for f in rng:
        P = []
        for k in range(count):
            if (k + 1) % 2 == 0:
                x = center[0] + r_list[f] * np.cos(angles[k])
                y = center[1] + r_list[f] * np.sin(angles[k])
                P.append((x, y))
            else:
                x = center[0] + np.flip(r_list)[f] * np.cos(angles[k])
                y = center[1] + np.flip(r_list)[f] * np.sin(angles[k])
                P.append((x, y))
        P = np.array(P)
        image = draw(img.copy(), color, P, t)
        frames.append([image])


if __name__ == '__main__':
    fig = plt.figure()
    N = 1024
    center = (N // 2, N // 2)
    img = np.zeros((N, N, 3), dtype=np.uint8) + 255
    # region optional
    count = 20
    R = 100
    t = np.linspace(0, 1, 1000)
    color = np.array([0, 0, 255], dtype=np.uint8)
    # endregion
    R_list = np.arange(0.5 * R, 1.5 * R + 1, 1)
    angles = np.linspace(2 * np.pi / count, 2 * np.pi, count)
    frames = []
    print('Step 1')
    frames_creation(img, color, frames, center, R_list, range(R_list.shape[0] // 2, R_list.shape[0]), t, angles)
    print('Step 2')
    frames_creation(img, color, frames, center, np.flip(R_list), range(R_list.shape[0]), t, angles)
    print('Step 3')
    frames_creation(img, color, frames, center, np.flip(R_list), np.flip(range(R_list.shape[0] // 2, R_list.shape[0])),
                    t,
                    angles)
    print('Frames creation finished')

    ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True, repeat_delay=0)
    writer = animation.PillowWriter(fps=24)
    ani.save('anim.gif', writer=writer)
    plt.show()

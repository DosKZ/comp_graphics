from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from lab5.coord_systems import rotation_xyz, orthogonal_projection, viewport


def normalize(vector):
    return vector / np.linalg.norm(vector) if np.linalg.norm(vector) != 0 else vector


def barycentric(point, x, y):
    u = np.cross([point[2][0] - point[0][0], point[1][0] - point[0][0], point[0][0] - x],
                 [point[2][1] - point[0][1], point[1][1] - point[0][1], point[0][1] - y], )
    if (abs(u[2]) < 1):
        return np.array([-1, -1, -1])
    else:
        return np.array([1. - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2]])


def raster(vertexes, faces, vertexes_normal, faces_normal, vertexes_texture, faces_texture, texture, img,
           with_texture=True, random_colors=False, plot=False):
    if with_texture and random_colors:
        print('params error!')
        return
    start = time()
    z_buff = np.ones(img.shape[:-1])
    width = texture.size[0]
    height = texture.size[1]
    texture_points = texture.load()
    o = [0, 0]

    for index in range(faces.shape[0]):
        face_points = vertexes[faces[index]]
        face_normals = vertexes_normal[faces_normal[index]]
        face_texture = vertexes_texture[faces_texture[index]]

        min_point = face_points[:, :-1].min(axis=0)
        # if index == 0:
        #     print(face_points[:, :-1].min(axis=0))
        #     print(face_points[:, :-1].max(axis=0))

        max_point = face_points[:, :-1].max(axis=0)
        # N=normalize(np.cross(face_points[2]-face_points[1],face_points[1]-face_points[0]))
        N = normalize(np.sum(face_normals, axis=0))
        back_face = np.dot([0,0,-1], N)
        # print(back_face)
        o[0] += 1
        if back_face > 0:
            o[1] += 1
            if random_colors:
                color = np.random.randint(0, 255, 3).astype(np.uint8)
            else:
                color = back_face * 255
            for x in range(min_point[0], max_point[0]):
                for y in range(min_point[1], max_point[1]):
                    if 0 < x < img.shape[0] and 0 < y < img.shape[1]:
                        # print(x,y)
                        bar_coords = barycentric(face_points[:, :-1], x, y)
                        if np.all(bar_coords >= 0):
                            z = np.dot(bar_coords, face_points[:, 2:])
                            if z < z_buff[x, y]:
                                if with_texture:
                                    u, v = np.dot(bar_coords, face_texture[:, :-1]) * [width, height]
                                    img[x, y] = np.array(texture_points[width - u, height - v])
                                else:
                                    img[x, y] = color
                                z_buff[x, y] = z
                    else:
                        continue
    stop = time()
    print(o)
    print(f'Отработал за {stop - start}')
    if plot:
        # img=plt.imshow(img,origin='lower')
        if with_texture:
            plt.imsave('results/Head_texture.jpg', img, origin='lower')
        elif random_colors:
            plt.imsave('results/Head_random_colors.jpg', img, origin='lower')
        else:
            plt.imsave('results/Head_grey.jpg', img, origin='lower')

        plt.imsave('results/ZBuffer.jpg', z_buff, origin='lower')
        plt.imshow(img, origin='lower')
        plt.show()
    else:
        return plt.imshow(img, origin='lower')


def bressenhem_draw(vertexes, faces, image=np.zeros((1024, 1024, 3), dtype=np.uint8),
                    color=np.array([255, 255, 255], dtype=np.uint8), plot=False):
    def line(img, clr, x0, y0, x1, y1):
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
        img[x_cur - 1, y_cur - 1] = clr
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
            img[x_cur - 1, y_cur - 1] = clr

    for first, second, third in faces:
        line(image, color, vertexes[first, 0], vertexes[first, 1], vertexes[second, 0],
             vertexes[second, 1])
        line(image, color, vertexes[second, 0], vertexes[second, 1], vertexes[third, 0],
             vertexes[third, 1])
        line(image, color, vertexes[first, 0], vertexes[first, 1], vertexes[third, 0],
             vertexes[third, 1])

    # image = np.rot90(image)
    if plot:
        plt.imsave('results/BHead.jpg', image, origin='lower')
        plt.imshow(image, origin='lower')
        plt.show()
    else:
        return plt.imshow(image, origin='lower')


def model_anim(frames_count, model_vertexes, faces, vertexes_normal, faces_normal, vertexes_texture, faces_texture, texture,
               img, viewport_params=[(0, 0), (1024, 1024)],
               with_texture=True, random_colors=False):
    angles_cam = np.linspace(0, 2 * np.pi, frames_count)
    frames = []
    fig = plt.figure()
    for i in range(frames_count):
        vrtx = model_vertexes.copy()
        vrtx_normal = vertexes_normal.copy()

        vrtx = (rotation_xyz([angles_cam[i], 0, 0]) @ vrtx.T).T
        vrtx_normal = (np.linalg.inv(rotation_xyz([angles_cam[i], 0, 0]).T) @ vrtx_normal.T).T

        vrtx = orthogonal_projection(vrtx)
        vrtx = viewport(vrtx, viewport_params[0], viewport_params[1])

        image = raster(vrtx, faces, vrtx_normal[:, :-1], faces_normal, vertexes_texture, faces_texture, texture,
                       img.copy(),
                       with_texture, random_colors)
        frames.append([image])
        print(f'frame #{i + 1}')
    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=0)
    writer = animation.PillowWriter(fps=24)
    ani.save('results/anim.gif', writer=writer)
    plt.show()

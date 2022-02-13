from PIL import Image

from coord_systems import *
from visual import raster, bressenhem_draw, model_anim


def to_projective(model_vertexes):
    return np.concatenate([model_vertexes, np.ones(model_vertexes.shape[0]).reshape(-1, 1)], axis=1)


def to_decart(model_vertexes):
    return (model_vertexes / model_vertexes[:, -1:])[:, :-1]


def parser(filename, mode='v'):
    f = open(filename, 'r').read()
    temp_v = []
    temp_f = []

    for line in f.split('\n'):
        index = -1
        if mode == 'v':
            if 'v ' in line:
                temp_v.append(line.split()[1:])
            elif 'f ' in line:
                index = 0
            else:
                continue
        elif mode == 'vt':
            if 'vt ' in line:
                temp_v.append(line.split()[1:])
            elif 'f ' in line:
                index = 1
            else:
                continue
        elif mode == 'vn':
            if 'vn ' in line:
                temp_v.append(line.split()[1:])
            elif 'f ' in line:
                index = 2
            else:
                continue
        else:
            print('Error!\n Wrong mode!')
            exit(1)
        if index != -1:
            temp_f.append(line.split()[1].split('/')[index])
            temp_f.append(line.split()[2].split('/')[index])
            temp_f.append(line.split()[3].split('/')[index])

    return np.array(temp_v, dtype=float), np.array(temp_f, dtype=int).reshape(-1, 3) - 1


if __name__ == '__main__':
    screen_size = 512
    img = np.zeros((screen_size, screen_size, 3), dtype=np.uint8)

    vertexes, faces = parser('african_head/african_head.obj', mode='v')
    vertexes_normal, faces_normal = parser('african_head/african_head.obj', mode='vn')
    vertexes_texture, faces_texture = parser('african_head/african_head.obj', mode='vt')
    texture = Image.open('african_head/african_head_diffuse.tga')

    print('Окееей... Летс гоу')

    angels = [np.pi / 1.5, -np.pi / 2.5, 0]
    transfer = [-1, 0, -1]
    scale = [0.9] * 3

    vertexes = to_projective(vertexes)
    vertexes_normal = to_projective(vertexes_normal)

    vertexes = mo2w(vertexes, angels, transfer, scale)
    vertexes_normal = mo2w(vertexes_normal, angels, transfer, scale, True)

    # [[cam_position],[obj_position]]
    positions = np.array([[2, 3, -2], [-2, -2, 0]])
    vertexes = mw2c(vertexes, positions[0], positions[1])
    vertexes_normal = mw2c(vertexes_normal, positions[0], positions[1], True)


    with_texture = False
    random_colors = False
    plot = True

    what = 0
    if what == 0:
        vertexes = orthogonal_projection(vertexes)
        vertexes = viewport(vertexes, (0, 0), (screen_size, screen_size))
        raster(vertexes, faces, vertexes_normal[:, :-1], faces_normal, vertexes_texture, faces_texture, texture,
               img.copy(),
               with_texture, random_colors, plot)
    elif what == 1:
        vertexes = orthogonal_projection(vertexes)
        vertexes = viewport(vertexes, (0, 0), (screen_size, screen_size))
        bressenhem_draw(vertexes, faces, img.copy(), plot=True)
    elif what == 2:
        frames_count = 20
        viewport_params = [(screen_size // 4, screen_size // 4), (screen_size // 2, screen_size // 2)]
        model_anim(frames_count, vertexes, faces, vertexes_normal, faces_normal, vertexes_texture, faces_texture,
                   texture,
                   img.copy(),
                   viewport_params, with_texture, random_colors)

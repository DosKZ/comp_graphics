import numpy as np
import matplotlib.pyplot as plt


def parser(filename, mode='v'):
    with open(filename, 'r') as f:
        f = f.read()
    lst = []
    for line in f.split('\n'):
        if 'v ' in line and mode == 'v':
            lst.append(line.split()[1:-1])
        if 'f ' in line and mode == 'f':
            lst.append(line.split()[1:])
    if mode == 'f':
        return np.array(lst, dtype=int) - 1
    else:
        return np.array(lst, dtype=float)


def size_convert(vertexes, size, scale):
    vertexes = (vertexes + scale) * size / (2 * scale)
    return np.rint(vertexes).astype(int)


def line(image, base_color, x0, y0, x1, y1):
    size=len(image)
    steep = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    signy = np.sign(y1 - y0)
    error = 0
    y = y0
    for x in range(x0, x1):
        color = base_color * (1 - np.sqrt((size/2 - x) ** 2 + (size/2 - y) ** 2) / size)
        if steep:
            image[y - 1, x - 1] = color
        else:
            image[x - 1, y - 1] = color
        error += dy
        if 2 * error >= dx:
            y += signy
            error -= dx


def draw(vertexes, faces, image=np.zeros((1024, 1024, 3), dtype=np.uint8),
         color=np.array([255, 255, 255], dtype=np.uint8)):
    for first, second, third in faces:
        line(image, color, vertexes[first, 0], vertexes[first, 1], vertexes[second, 0],
             vertexes[second, 1])
        line(image, color, vertexes[second, 0], vertexes[second, 1], vertexes[third, 0],
             vertexes[third, 1])
        line(image, color, vertexes[first, 0], vertexes[first, 1], vertexes[third, 0],
             vertexes[third, 1])

    plt.imsave("teapot.jpg", image)
    plt.imshow(image)
    plt.show()


screen_size = 1024
img = np.zeros((screen_size, screen_size, 3), dtype=np.uint8)+0
color = np.array([255, 192, 203], dtype=np.uint8)
scale = 5
# print(img)
#            Не трогать!
model_vertexes = size_convert(parser('teapot.obj', mode='v'), screen_size/2, scale)
faces = parser('teapot.obj', mode='f')

draw(model_vertexes, faces, img, color)

import  numpy as np
from math import sqrt
import cv2
import matplotlib.pyplot as plt


def rotation(img, deg):
    h, w, ch = img.shape
    ang = np.deg2rad(deg)

    center = (int(h / 2), int(w / 2))
    new_img = np.zeros((h, w, ch), dtype=img.dtype)

    for y in range(h):
        for x in range(w):
            yp = center[0] - y
            xp = center[1] - x

            rotz = [[np.cos(ang), -np.sin(ang)],
                    [np.sin(ang), np.cos(ang)]]

            new_coord = np.dot(rotz, [xp, yp])

            x_rot = (center[1] - new_coord[0]).astype(int)
            y_rot = (center[0] - new_coord[1]).astype(int)

            new_img[x_rot][y_rot][:] = img[x][y][:]

    return new_img

def flipX(img):
    h, w, ch = img.shape
    new_img = np.zeros((h, w, ch), dtype=img.dtype)

    for y in range(h):
        for x in range(w):
            flipx = [[1, 0],
                     [0, -1]]

            new_coord = np.dot(flipx, [x, y])
            new_img[new_coord[0]][new_coord[1]][:] = img[x][y][:]

    return new_img

def flipY(img):
    h, w, ch = img.shape
    new_img = np.zeros((h, w, ch), dtype=img.dtype)

    for y in range(h):
        for x in range(w):
            flipy = [[-1, 0],
                     [0, 1]]

            new_coord = np.dot(flipy, [x, y])
            new_img[new_coord[0]][new_coord[1]][:] = img[x][y][:]

    return new_img

def flipXY(img):
    return flipY(flipX(img))

def translation(img, ts):
    h, w, ch = img.shape
    tx, ty = ts[0], ts[1]

    new_x = w + abs(tx)
    new_y = h + abs(ty)
    output_img = np.zeros((new_y, new_x, ch), dtype=img.dtype)

    output_img[new_y - h:, new_x - w:] = img[:, :]

    return output_img

def scale(img, ss):
    h, w, _ = img.shape
    sx, sy = ss[0], ss[1]
    sh, sw = int(h * sy), int(w * sx)

    x_idx = np.arange(sw)
    y_idx = np.arange(sh)
    x, y = np.meshgrid(x_idx, y_idx)

    new_x = (x / sx).astype(int)
    new_y = (y / sy).astype(int)
    output_img = img[new_y, new_x]

    return output_img


if __name__ == '__main__':
    tests = {'rotation': (0, 90, 180, 270, 360),
             'translation': ((10, 10), (200, 200), (80, 0)),
             'scale': ((1, 1), (0.5, 0.5), (2, 0.1))
             }

    img = cv2.imread('images/logo_tec.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display images
    fig1, ax1 = plt.subplots(1, 5, figsize=(10, 10), sharex=True, sharey=True)
    fig1.suptitle('Rotaciones')
    for index, test in enumerate(tests['rotation']):
        img_rot = rotation(img, test)
        ax1[index].set_title("Grados: " + str(test))
        ax1[index].imshow(img_rot)

    fig2, ax2 = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
    fig2.suptitle('Flips')
    img_flip = flipX(img)
    ax2[0].set_title('Flip X')
    ax2[0].imshow(img_flip)
    img_flip = flipY(img)
    ax2[1].set_title('Flip Y')
    ax2[1].imshow(img_flip)
    img_flip = flipXY(img)
    ax2[2].set_title('Flip XY')
    ax2[2].imshow(img_flip)

    fig3, ax3 = plt.subplots(1, 3, figsize=(10, 10))
    fig3.suptitle('Traslaciones')
    for index, test in enumerate(tests['translation']):
        img_rot = translation(img, test)
        ax3[index].set_title("Traslación: " + str(test))
        ax3[index].imshow(img_rot)

    fig4, ax4 = plt.subplots(1, 3, figsize=(10, 10))
    fig4.suptitle('Escalaciones')
    for index, test in enumerate(tests['scale']):
        img_rot = scale(img, test)
        ax4[index].set_title("Escala: " + str(test))
        ax4[index].imshow(img_rot)

    fig1.savefig('images/Rotation')
    fig2.savefig('images/Flips')
    fig3.savefig('images/Traslation')
    fig4.savefig('images/Scale')

    plt.show()
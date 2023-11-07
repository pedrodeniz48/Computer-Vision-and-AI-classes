#Author: Pedro Deniz

import numpy as np
import matplotlib.pyplot as plt
import cv2


class KeypointGenerator:
    def __init__(self) -> None:
        self.localization = None
        self.scale = None
        self.orientation = None


class SIFT:
    def __init__(self, img) -> None:
        self._img = img
        self._img_gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        self._no_oct = 4
        self._octaves = []
        self._blur_level_init = 1.6
        self._scales = [1/i for i in [1, 2, 4, 16, 32]]
        self._scales_big = [1*i for i in [1, 2, 4, 16, 32]]
        self._keypoints = []

    def multiscale_extrem(self):

        for i in range(self._no_oct):
            octave_imgs = self._obtain_octaves(i)
            self._octaves.append(octave_imgs)

        dog_imgs = self._DOG(self._octaves)
        Kpts = self._find_cndtts_kpts(dog_imgs)

        return Kpts

    def _obtain_octaves(self, sigma_init):
        octave = []

        for i in range(len(self._scales)):
            dim = (int(self._img_gray.shape[1] * self._scales[i]), int(self._img_gray.shape[0] * self._scales[i]))
            scl_img = cv2.resize(self._img_gray, dim)
            sigma = 1.0 * self._blur_level_init ** (sigma_init + i)
            octave.append(cv2.GaussianBlur(scl_img, (3,3), sigmaX=sigma, sigmaY=sigma))

        return octave

    def _DOG(self, gauss_imgs):
        dog_img = []

        for octave in range(len(gauss_imgs) - 1):
            dog_img_oct = []
            for img in range(len(gauss_imgs[octave])):
                dog_img_oct.append(np.subtract(gauss_imgs[octave + 1][img], gauss_imgs[octave][img]))
            dog_img.append(dog_img_oct)

        return dog_img

    def _find_cndtts_kpts(self, DOG):

        keypts_cand = []

        for scale in range(len(DOG[0])):
            keypts_loc_sc = []

            dog_img_sup = DOG[0][scale]
            dog_img_act = DOG[1][scale]
            dog_img_inf = DOG[2][scale]

            height, width = dog_img_act.shape
            for h in range(1, height - 1):
                for w in range(1, width - 1):
                    eval_pix = dog_img_act[h, w]
                    if eval_pix != 0:
                        sup = dog_img_sup[(h - 1):(h + 2), (w - 1):(w + 2)]
                        act = dog_img_act[(h - 1):(h + 2), (w - 1):(w + 2)]
                        inf = dog_img_inf[(h - 1):(h + 2), (w - 1):(w + 2)]
                        act[1][1] = act[1][2]  # Letting value of 26 neighbors

                        pix_max = np.max([np.max(sup), np.max(act), np.max(inf)])
                        pix_min = np.min([np.min(sup), np.min(act), np.min(inf)])

                        if eval_pix > pix_max or eval_pix < pix_min:
                            keypts_loc_sc.append((h, w))

            keypts_cand.append(keypts_loc_sc)

        keypts_final = []

        for i in range(len(keypts_cand)):

            (r, c) = self._octaves[0][i].shape

            keypts_img = np.zeros((r, c), dtype=np.uint8)
            for j in range(len(keypts_cand[i])):
                x, y = keypts_cand[i][j]
                keypts_img[x, y] = 255

            if keypts_img.any():
                dim = (c * self._scales_big[i], r * self._scales_big[i])
                keypts_img = cv2.resize(keypts_img, dim)
                keypts_final.append(keypts_img)

        return np.array(keypts_final)

    def keypoints_localization(self, cand_pts):

        kpt_rfnd= []
        print(cand_pts.shape)
        scales, height, width = cand_pts.shape

        for scale in range(scales):
            for h in range(1, height - 1):
                for w in range(1, width - 1):
                    if cand_pts[scale, h, w] != 0:
                        candidate_x, candidate_y = w, h
                        neighborhood = self._img_gray[(h - 1):(h + 2), (w - 1):(w + 2)]

                        dx = (neighborhood[1, 2] - neighborhood[1, 0]) / 2.0
                        dy = (neighborhood[2, 1] - neighborhood[0, 1]) / 2.0
                        ds = (neighborhood[1, 1] - neighborhood[1, 1]) / 2.0

                        dxx = neighborhood[1, 2] - 2 * neighborhood[1, 1] + neighborhood[1, 0]
                        dyy = neighborhood[2, 1] - 2 * neighborhood[1, 1] + neighborhood[0, 1]
                        dss = neighborhood[1, 1] - 2 * neighborhood[1, 1] + neighborhood[1, 1]
                        dxy = (neighborhood[2, 2] - neighborhood[0, 2] - neighborhood[2, 0] + neighborhood[0, 0]) / 4.0
                        dxs = (neighborhood[1, 2] - neighborhood[1, 0] - neighborhood[1, 2] + neighborhood[1, 0]) / 4.0
                        dys = (neighborhood[2, 1] - neighborhood[0, 1] - neighborhood[2, 1] + neighborhood[0, 1]) / 4.0

                        delta = np.linalg.lstsq(np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]]),
                                                np.array([-dx, -dy, -ds]), rcond=None)[0]

                        refined_x = candidate_x + delta[0]
                        refined_y = candidate_y + delta[1]
                        k_pnt = KeypointGenerator()
                        k_pnt.localization = (int(refined_y), int(refined_x))
                        k_pnt.scale = (self._scales[scale])

                        kpt_rfnd.append((refined_y, refined_x))
                        self._keypoints.append(k_pnt)

        (r, c) = self._img_gray.shape
        temp_kpts = []
        keypts_img = np.zeros((r, c), dtype=np.uint8)
        for k in range(0, len(self._keypoints), 50):
            y, x = self._keypoints[k].localization
            try:
                keypts_img[y, x] = 255
                temp_kpts.append(self._keypoints[k])
            except:
                pass

        self._keypoints = temp_kpts

    def assign_orientation(self):

        grad_x = cv2.Sobel(self._img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self._img_gray, cv2.CV_64F, 0, 1, ksize=3)
        orientations = np.arctan2(grad_y, grad_x)

        temp_keypoints = []
        for index, point in enumerate(self._keypoints):
            y, x = point.localization

            if x > 0 and y > 0:
                try:
                    point.orientation = orientations[y, x]
                    temp_keypoints.append(point)
                except:
                    pass

        self._keypoints = temp_keypoints

    def descriptors_generation(self):

        descriptors = []

        for point in self._keypoints:
            y, x = point.localization

            neighborhood_size = 16
            sub_block_size = 4

            angle = point.orientation

            if y > (self._img_gray.shape[0] - 7) or x > (self._img_gray.shape[1] - 7):
                continue

            neighborhood = self._img_gray[int(y - 8):int(y + 7), int(x - 8):int(x + 7)]

            if not neighborhood.any():
                continue

            descriptor = np.zeros(128)

            for i in range(neighborhood_size):
                sub_block_x = i % (neighborhood_size // sub_block_size)
                sub_block_y = i // (neighborhood_size // sub_block_size)

                sub_block = neighborhood[sub_block_y * sub_block_size:(sub_block_y + 1) * sub_block_size,
                            sub_block_x * sub_block_size:(sub_block_x + 1) * sub_block_size]

                sub_block_grad_x = cv2.Sobel(sub_block, cv2.CV_64F, 1, 0, ksize=3)
                sub_block_grad_y = cv2.Sobel(sub_block, cv2.CV_64F, 0, 1, ksize=3)
                sub_block_orientation = np.arctan2(sub_block_grad_y, sub_block_grad_x)

                sub_block_orientation -= angle
                sub_block_orientation[sub_block_orientation < 0] += 2 * np.pi

                histogram = np.histogram(sub_block_orientation, bins=8, range=(0, 2 * np.pi))[0]

                descriptor[i * 8:(i + 1) * 8] = histogram

                descriptor /= np.linalg.norm(descriptor)

            descriptors.append(descriptor)

        return descriptors

    def draw_keypts(self, img):
        for keypoint in self._keypoints:
            y, x = keypoint.localization
            scale = keypoint.scale
            ang = keypoint.orientation

            radius = int(20 * scale)
            center = (x, y)
            end = (int(x + radius * np.cos(ang)), int(y + radius * np.sin(ang)))
            color = (200, 0, 200)
            thick = 1

            cv2.circle(img, center, radius, color, thick)
            cv2.line(img, center, end, color, thick)

if __name__ == '__main__':
    img = cv2.imread('images/eiad.jpeg')
    r, c, _ = img.shape
    cv2.imshow("Original", img)

    sift1 = SIFT(img)
    candidates_keypoints = sift1.multiscale_extrem()
    sift1.keypoints_localization(candidates_keypoints)
    sift1.assign_orientation()
    descriptors = sift1.descriptors_generation()

    sift1.draw_keypts(img)
    cv2.imshow(f'SIFT', img)
    cv2.imwrite('images/eiad_SIFT.png', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
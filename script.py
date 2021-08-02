import math
import random

import cv2
import numpy
import numpy as np

# Q3 part1
import script2


def part_one():
    dylan = cv2.imread("./inputs/Dylan.jpg")
    [r, c, x] = dylan.shape
    quads = cv2.imread("./inputs/frames.jpg")
    [r2, c2, x] = quads.shape

    source_xs_affine = np.float32([[0, 0], [c, 0], [0, r]])
    source_xs_projective = np.float32([[0, 0], [c, 0], [c, r], [0, r]])

    proj_pts = np.float32([[196, 56], [495, 160], [431, 498], [38, 184]])
    affine_pts = np.float32([[552, 220], [846, 68], [607, 454]])

    M = cv2.getAffineTransform(source_xs_affine, affine_pts)
    M2 = cv2.getPerspectiveTransform(source_xs_projective, proj_pts)

    dst = cv2.warpAffine(dylan, M, (c2, r2), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 255, 0))
    dst2 = cv2.warpPerspective(dylan, M2, (c2, r2), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 255, 0))

    images = [dst, dst2]

    for image in images:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image[i, j]
                rgb = (pixel[0], pixel[1], pixel[2])
                if rgb != (0, 255, 0) and tuple(quads[i, j]) != (0, 0, 0):
                    # print(rgb)
                    quads[i, j] = image[i, j]
                pass

    cv2.imshow("dylan", quads)
    cv2.waitKey(0)


# Q3 part 2
def part_two(iterations, img1, img2, threshold, projective=False):
    if projective:
        k = 4
        transform = cv2.getPerspectiveTransform
        warp = cv2.warpPerspective
    else:
        k = 3
        transform = cv2.getAffineTransform
        warp = cv2.warpAffine

    kp1, kp2, good = script2.part_two(img1, img2)

    # find matches from both images from Q2
    # compute affin transformation by randomly select group of 3 matches

    best_model_inliers = -1
    best_M = None

    for i in range(iterations):
        counter = 0
        random_matches = random.sample(good, k)  # Sample random matches according to homographic/affine parameter = k
        sources = np.float32([kp1[random_matches[i][0].queryIdx].pt for i in range(k)])
        targets = np.float32([kp2[random_matches[i][0].trainIdx].pt for i in range(k)])

        M = transform(sources, targets)

        for m in good:
            m_source = kp1[m.queryIdx].pt
            m_target_old = kp2[m.trainIdx].pt
            m_target_new = numpy.matmul(M, [m_source[0], m_source[1], 1])
            # print(m_target_new)
            if projective and m_target_new[2] != 0:
                m_target_new[0] = m_target_new[0] / m_target_new[2]
                m_target_new[1] = m_target_new[1] / m_target_new[2]

            euclidean_distance = math.sqrt(
                ((m_target_old[0] - m_target_new[0]) ** 2) + ((m_target_old[1] - m_target_new[1]) ** 2))
            if euclidean_distance < threshold:
                counter += 1

        if counter > best_model_inliers:
            best_model_inliers = counter
            best_M = M

    [r, c] = img2.shape[:2]
    dst = warp(img1, best_M, (round(1.3 * c), round(1.3 * r)), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    for i in range(r):
        for j in range(c):
            p = tuple(img2[i, j])
            if p != (0, 0, 0):
                dst[i, j] = img2[i, j]

    print(best_M)
    cv2.imshow("Result", dst)


def run_script():
    # part_one()
    img1 = cv2.imread("./inputs/pair3_imageA.jpg")
    img2 = cv2.imread("./inputs/pair3_imageB.jpg")
    # part_two(img1, img2, False)
    part_two(200, img1, img2, True)


if __name__ == "__main__":
    run_script()

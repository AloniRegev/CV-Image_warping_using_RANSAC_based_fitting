import math
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawMatches(img, kp):
    # similar to opencv's feature drawing, except only one color is used

    for i in range(len(kp)):
        curr = kp[i]
        p = (round(curr.pt[0]), round(curr.pt[1]))
        scale = curr.size
        orientation = curr.angle
        img = cv2.circle(img, p, round(scale), color=(0,255,0))
        new_p_x = round(scale*math.cos(orientation)) + p[0]
        new_p_y = round(scale*math.sin(orientation)) + p[1]
        img = cv2.line(img, p, (new_p_x, new_p_y),color=(0,255,0))
    return img

def drawMatches2Images(img1, img2, kp1, kp2, good):
    # connects each pair of matches in "good".
    [r, c] = img1.shape[:2]
    new_img = cv2.hconcat([img1, img2])
    cv2.waitKey(0)
    for i in range(min(len(good),50)):
        curr_match = good[i]
        first_p = kp1[curr_match.queryIdx].pt
        second_p = kp2[curr_match.trainIdx].pt
        first_p = (round(first_p[0]), round(first_p[1]))
        new_second_p = (round(second_p[0]+c), round(second_p[1]))
        color = (random.randint(0,256),random.randint(0,256),random.randint(0,256))
        new_img = cv2.line(new_img, first_p, new_second_p,color, thickness=1)
    return new_img

def part_one():
    sift = cv2.SIFT_create()

    uoh = cv2.imread("./Q2/UoH.jpg")
    kp3, des3 = sift.detectAndCompute(uoh, None)
    random_kp3 = random.sample(kp3, 500)
    img = drawMatches(uoh, random_kp3)
    cv2.imshow("UOH", img)
    cv2.waitKey(0)


def part_two(img1, img2, bi=False):

    sift = cv2.SIFT_create()
    # img1 = cv2.imread('./Q2/pair1_imageA.jpg')
    # img2 = cv2.imread('./Q2/pair1_imageB.jpg')
    grey_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grey_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(grey_img1, None)
    kp2, des2 = sift.detectAndCompute(grey_img2, None)

    # Calculate matches
    matches = knn_match(des1, des2, k=2)
    matches2 = knn_match(des2, des1, k=2)


    good = []
    if not bi:
        for m, n in matches:
            if m.distance < 0.8*n.distance:
                good.append(m)
    else:
        for m1, m11 in matches:
            for m2, m22 in matches2:
                if kp1[m1.queryIdx] == kp1[m2.trainIdx] and kp2[m1.trainIdx] == kp2[m2.queryIdx]:
                    good.append(m1)

    img3 = drawMatches2Images(img1, img2, kp1, kp2, good)
    plt.imshow(img3),plt.show()
    return kp1, kp2, good



def descriptor_distance(des1, des2):
    return np.linalg.norm(des1 -des2)

def calc_distance_matrix(des1, des2):
    list = []
    for i in range(len(des1)):
        curr = []
        for j in range(len(des2)):
            curr.append(descriptor_distance(des1[i], des2[j]))
        list.append(curr)

    return list


class match:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance

def knn_match(des1, des2, k=2):

    distance_matrix = calc_distance_matrix(des1, des2)
    toRet = []
    for i in range(len(des1)):
        x = distance_matrix[i]
        dict = {}
        for j in range(len(x)):
            dict[j] = x[j]
        sorted_dict = {k: v for k,v in sorted(dict.items(), key = lambda x: x[1])}
        list = []
        k_counter = 0
        for t in sorted_dict:
            if k_counter == k:
                break
            list.append(match(i,t,sorted_dict[t]))
            k_counter += 1
        toRet.append(list)
    return toRet


def run_script():
    img1 = cv2.imread("./Q2/pair2_imageA.jpg")
    img2 = cv2.imread("./Q2/pair2_imageB.jpg")
    #part_two(img1, img2, False)
    part_two(img1, img2, True)

if __name__ == "__main__":
    run_script()
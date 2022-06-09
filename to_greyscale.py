import cv2
import os
import numpy as np


def main(type):
    for image in os.listdir(f"/home/aidan/fiftyone/coco/{type}2017"):
        imgGray = cv2.imread(f"/home/aidan/fiftyone/coco/{type}2017/{image}", 0)

        cv2.imwrite(f"/home/aidan/fiftyone/coco (copy)/{type}2017/{image}", imgGray)


if __name__ == '__main__':
    main('val')
    main('train')

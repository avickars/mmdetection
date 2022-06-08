import cv2
import os


def main(type):
    # imgGray = cv2.imread('test.jpg', 0)

    for image in os.listdir(f"/home/aidan/fiftyone/coco (copy)/{type}2017"):
        imgGray = cv2.imread(f"/home/aidan/fiftyone/coco (copy)/{type}2017/{image}", 0)

        cv2.imwrite(f"/home/aidan/fiftyone/coco (copy)/{type}2017-grey/{image}", imgGray)


if __name__ == '__main__':
    main('val')
    main('train')

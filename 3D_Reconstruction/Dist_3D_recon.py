import numpy as np
from skimage import io, transform, color
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import os
import cv2

def readImages(inDir):
    image_list = os.listdir(inDir)
    image_list.sort()

    images = []
    for image_file in image_list:
        image_path = os.path.join(inDir, image_file)
        image = cv2.imread(image_path)

        images.append(image)

    return images

def StackRotate(images, tiltAngle):
    rotated = []
    for img in images:
        h, w = img.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), tiltAngle, 1.0)
        rotated_image = cv2.warpAffine(img, rotationMatrix, (w, h))
        rotated.append(rotated_image)

    return rotated

def MarkPosition(image):
    # fig, ax = plt.subplots()
    # ax.imshow(image, cmap="gray")
    # clicker=plt.ginput(1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3)
    # plt.close(fig)

    # return int(clicker[0][1])
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    klicker = clicker(ax, ["Mark"], markers=["x"])
    plt.show()
    position = int(klicker.get_positions()["Mark"][0][1])

    return position

def StackVerticalShift(images):
    firstImage = images[0]
    lastImage = images[-1]
    
    firstPos = MarkPosition(firstImage)
    secondPos = MarkPosition(lastImage)
    shift = abs(secondPos - firstPos)/len(images)

    shifted = []
    for count, img in enumerate(images):
        translation = transform.SimilarityTransform(translation=(0, -2*shift*count))
        shiftedImg = transform.warp(img, translation)
        shiftedImg = (shiftedImg*255).astype(np.uint8)
        shifted.append(shiftedImg)
    # fig, ax = plt.subplots()
    # ax.imshow(firstImage, cmap="gray")
    # klicker = clicker(ax, ["Mark"], markers=["x"])
    # plt.show()
    # firstPos = int(klicker.get_positions()["Mark"][0][1])

    # fig2, ax2 = plt.subplots()
    # ax2.imshow(lastImage, cmap="gray")
    # klicker2 = clicker(ax2, ["Mark"], markers=["x"])
    # plt.show()
    # secondPos = int(klicker2.get_positions()["Mark"][0][1])

    # shift = (abs(secondPos-firstPos)) / len(images)

    # count = 0
    # shifted = []
    # for img in images:
    #     transformation = transform.EuclideanTransform(translation = [0, (-1*shift*count)])
    #     img = transform.warp(img, transformation)
    #     img *= 255
    #     img = np.array(img, dtype=np.uint8)
    #     shifted.append(img)
    #     count += 1

    return shifted

images = readImages("C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/MOX_Slices/Stack1_B1")
rotated = StackRotate(images=images, tiltAngle=-8.23)
shifted = StackVerticalShift(images=rotated)

cv2.imshow('Concat Image', cv2.resize(np.hstack((shifted[0],shifted[-1])), (1600, 600)))  # Resize the image for display
cv2.waitKey(0)
cv2.destroyAllWindows()
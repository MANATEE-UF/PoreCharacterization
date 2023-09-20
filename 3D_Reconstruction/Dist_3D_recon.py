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
    for image in images:
        h, w = image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), tiltAngle, 1.0)
        rotated_image = cv2.warpAffine(image, rotationMatrix, (w, h))
        rotated.append(rotated_image)

    return rotated

def GetYCoord(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    klicker = clicker(ax, ["Mark"], markers=["x"])
    plt.show()
    position = int(klicker.get_positions()["Mark"][0][1])

    return position

    # fig, ax = plt.subplots()
    # ax.imshow(image, cmap="gray")
    # clicker=plt.ginput(1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3)
    # plt.close(fig)

    # return int(clicker[0][1])

def StackVerticalShift(images):
    firstImage = images[0]
    lastImage = images[-1]
    
    firstPos = GetYCoord(firstImage)
    secondPos = GetYCoord(lastImage)
    shift = abs(secondPos - firstPos)/len(images)

    shifted = []
    for count, image in enumerate(images):
        translation = transform.SimilarityTransform(translation=(0, -1*shift*count))
        # translation = transform.EuclideanTransform(translation = [0, (-1*shift*count)])
        shiftedImage = transform.warp(image, translation)
        shiftedImage = (shiftedImage*255).astype(np.uint8)
        shifted.append(shiftedImage)
    
    return shifted

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

    # return shifted

def GetCoords(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    klicker = clicker(ax, ["Mark"], markers=["x"])
    plt.show()
    xCoord, yCoord = int(klicker.get_positions()["Mark"][0][0]), int(klicker.get_positions()["Mark"][0][1])
    
    return xCoord, yCoord

#This could definitely be improved, maybe drawing a bounding box and fining the top left and bottom right coords of it
def StackCropping(images):
    image = images[0]

    x1, y1 = GetCoords(image) #Select top-left corner of AOI
    x2, y2 = GetCoords(image) #Select bottom-right corner of AOI

    cropped = []
    for image in images:
        cropped_image = image[y1:y2, x1:x2]
        cropped.append(cropped_image)

    return cropped

def main():
    inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Unprocessed/Stack1_B1"
    outDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Processed/Stack1_B1"

    images = readImages(inDir)
    rotated = StackRotate(images, tiltAngle=-8.23)
    shifted = StackVerticalShift(rotated)
    cropped = StackCropping(shifted)

    cv2.imshow('Concat Image', cv2.resize(np.hstack((cropped[0],cropped[-1])), (1600, 600)))  # Resize the image for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import numpy as np
from skimage import io, transform, color
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import os


def Translate(image, xTransform, yTransform):
    tr = transform.EuclideanTransform(translation=[xTransform, yTransform])
    img = transform.warp(image, tr)
    return img

def StackVerticalShift(stackFolder):
    images = os.listdir(stackFolder)
    images.sort()

    firstImage = stackFolder + "/" + images[0]
    lastImage = stackFolder + "/" + images[-1]

    firstImage = io.imread(firstImage)
    lastImage = io.imread(lastImage)
    
    fig, ax = plt.subplots()
    ax.imshow(firstImage, cmap="gray")
    klicker = clicker(ax, ["Mark"], markers=["x"])
    plt.show()
    firstPos = int(klicker.get_positions()["Mark"][0][1])

    fig2, ax2 = plt.subplots()
    ax2.imshow(lastImage, cmap="gray")
    klicker2 = clicker(ax2, ["Mark"], markers=["x"])
    plt.show()
    secondPos = int(klicker2.get_positions()["Mark"][0][1])

    shift = (abs(secondPos-firstPos)) / len(images)

    count = 0
    for image in images:
        image = stackFolder + "/" + image
        img = io.imread(image)
        img2 = io.imread(image)
        img = Translate(img, 0, -1 * shift * count)
        img *= 255
        img = np.array(img, dtype=np.uint8)
        io.imsave(f"{count}.tif", img)
        count += 1

def SplitStackIntoQuadrants(image):
    pass

def SplitImageIntoQuadrants(image):
    pass

def AugmentStackData(stackFolder):

def main():
    StackVerticalShift("/Users/mitchellmika/Desktop/TrainingData")

if __name__ == "__main__":
    main()


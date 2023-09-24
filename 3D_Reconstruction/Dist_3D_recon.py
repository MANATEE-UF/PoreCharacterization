import numpy as np
from skimage import io, transform, color
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import os
import cv2
from scipy.signal import wiener
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from matplotlib.widgets import RectangleSelector

def readImages(inDir):
    image_list = os.listdir(inDir)
    image_list.sort()

    images = []
    for image_file in image_list:
        if image_file.endswith(".tif"):
            image_path = os.path.join(inDir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
        else:
            print(f"Removed File {image_file} (Returned None)")

    return images

def StackRotate(images, tiltAngle):
    rotated = []

    for i, image in enumerate(images):
        if image is not None:
            h, w = image.shape[:2]
            rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), tiltAngle, 1.0)
            rotated_image = cv2.warpAffine(image, rotationMatrix, (w, h))
            rotated.append(rotated_image)
        else:
            print("Warning: Skipping image at index {} (Returned None)".format(i))

    return rotated

def GetYCoord(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    klicker = clicker(ax, ["Mark"], markers=["x"])
    plt.show()
    
    positions = klicker.get_positions()

    if "Mark" in positions:
        yCoord = int(positions["Mark"][0][1])

        return yCoord
    else:
        print("Please select only one mark on the image.")
        return None

#Doesn't seem too shift well, may be good to add loop that increases shift until user says otherwise
def StackVerticalShift(rotated, shift=None):
    firstImage = rotated[0]
    lastImage = rotated[-1]
    
    if shift == None:
        print("Pick a point that will be visible in the final image slice")
        firstPos = GetYCoord(firstImage)
        print("Pick that same point on the final image slice")
        secondPos = GetYCoord(lastImage)
        shift = abs(secondPos - firstPos)/len(rotated)
        
    shifted = []
    for count, image in enumerate(rotated):
        translation = transform.SimilarityTransform(translation=(0, -1*shift*count))
        shiftedImage = transform.warp(image, translation)
        shiftedImage = (shiftedImage*255).astype(np.uint8)
        shifted.append(shiftedImage)
    
    return shifted

def GetRectangleCoords(image):
    def onselect(eclick, erelease):
        global xmin, xmax, ymin, ymax
        xmin, xmax = sorted([int(eclick.xdata), int(erelease.xdata)])
        ymin, ymax = sorted([int(eclick.ydata), int(erelease.ydata)])
    
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    RS = RectangleSelector(ax, onselect, useblit=True)    
    plt.show()

    return xmin, ymin, xmax, ymax

def StackCropping(shifted):
    shifted1st = shifted[0]

    print("Select top-left and bottom-right corners of rectangular AOI")
    x1, y1, x2, y2 = GetRectangleCoords(shifted1st) #Select top-left and bottom-right corner of rectangular AOI

    cropped = []
    for image in shifted:
        cropped_image = image[y1:y2, x1:x2]
        cropped.append(cropped_image)

    return cropped

def NoiseReduction(cropped):
    filtered = []

    for i in range(len(cropped)):
        img = cropped[i]   
        img = img/255
        filteredImg = wiener(img, (5,5))
        filteredImg = (filteredImg * 255).astype(np.uint8)
        filtered.append(filteredImg)

    return filtered

def CreateRectangleMask(image):
    h, w = image.shape[:2]
    remove = np.zeros((h, w), dtype=np.uint8)

    print("Select top-left (blue x) and bottom-right (red x) corners of 1st rectangular AOI")
    R1x1, R1y1, R1x2, R1y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Left Rectangle
   
    print("Select top-left (blue x) and bottom-right (red x) corners of 2nd rectangular AOI")
    R2x1, R2y1, R2x2, R2y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Right Rectangle

    # Rectangles should extand to outer edges, hard to do so with GetCoords() so this extends the correct cordinates for each rectangle
    if R1x1 != 0:
        R1x1 = 0
    
    if R2x2 != w - 1:
        R2x2 = w - 1
    
    remove[R1y1:R1y2+1, R1x1:R1x2+1] = 1
    remove[R2y1:R2y2+1, R2x1:R2x2+1] = 1

    cv2.imshow('Binary Mask', remove)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    return remove

# NOT WORKING
def FFT_Filtering(filtered):
    applyFilterAgain = True

    while applyFilterAgain is True:
        img = np.empty_like(filtered, dtype=complex)
        img[0] = fftshift(fft2(filtered[0]))
        imgAbsLog = np.log(1+np.abs(img[0]))
        
        remove = CreateRectangleMask(imgAbsLog)
        print(remove.shape)

        maskedImg = np.empty_like(img, dtype=complex)
        maskedImg[0] = img[0] - (img[0] * remove)

        fftFiltered = np.empty_like(filtered, dtype=complex)
        fftFiltered[0] = ifft2(ifftshift(maskedImg[0]))

        plt.figure(figsize=(8, 6))

        plt.subplot(2, 1, 1)
        plt.imshow(filtered[0], cmap='gray')
        plt.title('Original Image')

        plt.subplot(2, 1, 2)
        plt.imshow(np.abs(fftFiltered[0]), cmap='gray')
        plt.title('Filtered Image')
        
        plt.tight_layout()
        plt.show()

        while True:
            ans = input("Apply FFT filter again? [Y/N]: ")

            if ans == "Y":
                applyFilterAgain = True
                break
            elif ans == "N":
                applyFilterAgain = False
                break
            else:
                print("Invalid Input")
    
    for i in range(2, len(filtered)):
        img[i] = fftshift(fft2(filtered[i]))

    for i in range(2, len(filtered)):
        maskedImg[i] = img[i] - (img[i] * remove)
        fftFiltered[i] = ifft2(ifftshift(maskedImg[i]))
    
    realImages = np.real(fftFiltered)
    min = np.min(realImages)
    max = np.max(realImages)

    normalizedRealImages = (realImages - min) / (max - min) * 255
    fftFiltered = normalizedRealImages.astype(np.uint8)

    return fftFiltered

def CheckArray(array):
    if isinstance(array, np.ndarray):
        print("It's a NumPy array!")
    else:
        print("It's not a NumPy array.")


def main():
    inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Unprocessed/Stack3_D"
    outDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Processed/Stack3_D"

    images = readImages(inDir)
    rotated = StackRotate(images, tiltAngle=-3.5)
    shifted = StackVerticalShift(rotated)
    cropped = StackCropping(shifted)
    filtered = NoiseReduction(cropped)
    fftFiltered = FFT_Filtering(filtered)
    cv2.imshow('FFT Filtered image', fftFiltered[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows

if __name__ == "__main__":
    main()
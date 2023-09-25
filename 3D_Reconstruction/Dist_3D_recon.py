import numpy as np
from skimage import io, transform, filters, morphology, color, exposure
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import os
import cv2
from scipy.signal import wiener
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from matplotlib.widgets import RectangleSelector
from PIL import Image

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

    if saveImages is True:
        SaveImages(rotated, "Rotated")

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

def StackVerticalShift(rotated, shift=None):
    if shift == None:
        shiftAgain = True

        firstImage = rotated[0]
        lastImage = rotated[-1]

        print("Pick a point that will be visible in the final image slice")
        firstPos = GetYCoord(firstImage)
        print("Pick that same point on the final image slice")
        secondPos = GetYCoord(lastImage)
        shiftNumerator = abs(secondPos - firstPos)
        shiftDenominator = len(rotated)

        while shiftAgain == True:
            shift = shiftNumerator/shiftDenominator

            shifted = []
            for count, image in enumerate(rotated):
                translation = transform.SimilarityTransform(translation=(0, -1*shift*count))
                shiftedImage = transform.warp(image, translation)
                shiftedImage = (shiftedImage*255).astype(np.uint8)
                shifted.append(shiftedImage)

            plt.figure(figsize=(8, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(shifted[0], cmap='gray')
            plt.title('First Slice')

            plt.subplot(1, 2, 2)
            plt.imshow(shifted[-1], cmap='gray')
            plt.title('Final Slice')
            
            plt.tight_layout()
            plt.show()

            while True:
                ans = input("Apply bigger shift [Y/N] or use previous shift margin [R]: ")

                if ans == "Y":
                    shiftDenominator -= 5
                    shiftAgain = True
                    break

                elif ans == "N":
                    print(f"Best shift margin is {shift}")
                    shiftAgain = False
                    break

                elif ans == "R":
                    shiftDenominator += 5
                    shift = shiftNumerator/shiftDenominator

                    shifted = []
                    for count, image in enumerate(rotated):
                        translation = transform.SimilarityTransform(translation=(0, -1*shift*count))
                        shiftedImage = transform.warp(image, translation)
                        shiftedImage = (shiftedImage*255).astype(np.uint8)
                        shifted.append(shiftedImage)
                    
                    print(f"Best shift margin is {shift}")
                    shiftAgain = False
                    break

                else:
                    print("Invalid Input")
    else:
        shifted = []
        for count, image in enumerate(rotated):
            translation = transform.SimilarityTransform(translation=(0, -1*shift*count))
            shiftedImage = transform.warp(image, translation)
            shiftedImage = (shiftedImage*255).astype(np.uint8)
            shifted.append(shiftedImage)
    
    if saveImages is True:
        SaveImages(shifted, "Shifted")

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

    print("Select rectangular AOI")
    x1, y1, x2, y2 = GetRectangleCoords(shifted1st) #Select top-left and bottom-right corner of rectangular AOI

    cropped = []
    for image in shifted:
        cropped_image = image[y1:y2, x1:x2]
        cropped.append(cropped_image)

    if saveImages is True:
        SaveImages(cropped, "Cropped")

    return cropped

def NoiseReduction(cropped):
    filtered = []

    for i in range(len(cropped)):
        img = cropped[i]   
        img = img/255
        filteredImg = wiener(img, (5,5))
        filteredImg = (filteredImg * 255).astype(np.uint8)
        filtered.append(filteredImg)

    if saveImages is True:
        SaveImages(filtered, "Filtered")

    return filtered

def CreateRectangleMask(image):
    h, w = image.shape[:2]
    remove = np.zeros((h, w), dtype=np.uint8)

    print("Select 1st rectangular AOI")
    R1x1, R1y1, R1x2, R1y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Left Rectangle
   
    print("Select 2nd rectangular AOI")
    R2x1, R2y1, R2x2, R2y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Right Rectangle

    # Rectangles should extand to outer edges, hard to do so with GetCoords() so this extends the correct cordinates for each rectangle
    if R1x1 != 0:
        R1x1 = 0
    
    if R2x2 != w - 1:
        R2x2 = w - 1
    
    remove[R1y1:R1y2+1, R1x1:R1x2+1] = 1
    remove[R2y1:R2y2+1, R2x1:R2x2+1] = 1
    return remove

#Not saving 2nd image of fftFiltered
def FFT_Filtering(filtered):
    applyFilterAgain = True

    while applyFilterAgain is True:
        img = np.empty_like(filtered, dtype=complex)
        img[0] = fftshift(fft2(filtered[0]))
        imgAbsLog = np.log(1+np.abs(img[0]))
        
        remove = CreateRectangleMask(imgAbsLog)

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

    if saveImages is True:
        SaveImages(fftFiltered, "FFTFiltered")

    return fftFiltered

def CheckArray(array):
    if isinstance(array, np.ndarray):
        print("It's a NumPy array!")
    else:
        print("It's not a NumPy array.")

def SaveImages(images, folderName):
    if not os.path.exists(outDir):
        print("Invalid Directory")
    else:
        folderPath = os.path.join(outDir, folderName)    
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
        # images = np.array(images).astype(np.uint8)

        for i, image in enumerate(images):
            fileName = f"{folderName}_{i}.tif"
            img = Image.fromarray(image, 'L')
            img.save(os.path.join(folderPath, fileName))

def DetermineThreshold(fftFiltered, deltaZ, nmPerPixel):
    fftFiltered = np.array(fftFiltered)

    smooth = filters.gaussian(fftFiltered, sigma=0.5)

    dims = smooth.shape
    newDims = (dims[0], dims[1], round(dims[2]*deltaZ))

    thresh = filters.threshold_otsu(smooth)

    
    voids = smooth < thresh #Use > for light selection (EDS). Use < for dark selection (bubbles)

def main():
    global inDir, outDir, saveImages
    inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Unprocessed/Stack3_D"
    outDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/ProcessedPython/Stack3_D"
    saveImages = False

    images = readImages(inDir)
    rotated = StackRotate(images, tiltAngle=-3.5)
    shifted = StackVerticalShift(rotated, shift=0.3497942386831276) #Enter + value for shift if shift margin is known
    cropped = StackCropping(shifted)
    filtered = NoiseReduction(cropped)
    fftFiltered = FFT_Filtering(filtered)
    cv2.imshow('FFT Filtered image', fftFiltered[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows

if __name__ == "__main__":
    main()
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import os
import cv2
from scipy.signal import wiener
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from matplotlib.widgets import RectangleSelector
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from skimage.filters import gaussian, threshold_multiotsu, threshold_otsu
from skimage.morphology import opening, closing, disk, ball
from skimage.segmentation import active_contour, clear_border
from tqdm import tqdm

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

#May want to consider making mask boolean array (should work the same as it currently does, just more efficient)
def CreateDoubleRectangleMask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    print("Select 1st rectangular AOI")
    R1x1, R1y1, R1x2, R1y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Left Rectangle
   
    print("Select 2nd rectangular AOI")
    R2x1, R2y1, R2x2, R2y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Right Rectangle

    # Rectangles should extand to outer edges, hard to do so with GetCoords() so this extends the correct cordinates for each rectangle
    if R1x1 != 0:
        R1x1 = 0
    
    if R2x2 != w - 1:
        R2x2 = w - 1
    
    mask[R1y1:R1y2+1, R1x1:R1x2+1] = 1
    mask[R2y1:R2y2+1, R2x1:R2x2+1] = 1
    return mask

def FFT_Filtering(filtered):
    applyFilterAgain = True

    while applyFilterAgain is True:
        img = np.empty_like(filtered, dtype=complex)
        img[0] = fftshift(fft2(filtered[0]))
        imgAbsLog = np.log(1+np.abs(img[0]))
        
        remove = CreateDoubleRectangleMask(imgAbsLog)

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
    
    for i in range(1, len(filtered)):
        img[i] = fftshift(fft2(filtered[i]))

    for i in range(1, len(filtered)):
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

def Imover(inputImage, mask, color=[255,255,255]):
    mask = mask != 0

    formattedImage = np.uint8(inputImage)

    if formattedImage.dim == 2:
        #Input is grayscale. Initialize all output channels the same
        outRed = formattedImage
        outGreen = formattedImage
        outBlue = formattedImage
    else:
        #Input is RGB truecolor
        outRed = formattedImage[:,:,0]
        outGreen = formattedImage[:,:,1]
        outBlue = formattedImage[:,:,2]

    outRed[mask] = np.round(color[0]).astype(np.uint8) + outRed[mask]
    outGreen[mask] = np.round(color[1]).astype(np.uint8) + outGreen[mask]
    outBlue[mask] = np.round(color[2]).astype(np.uint8) + outBlue[mask]

    out = cv2.merge([outBlue, outGreen, outRed])
    
    return out

def ThreshFeedback(img, lowThresh, highThresh, name):
    slice = img[:, :, round(img.shape[2] / 5)]
    slice = np.pad(slice, ((0, img.shape[0]), (0, img.shape[1])), mode='constant', constant_values=0)
    slice_BW = np.logical_and(slice > lowThresh, slice < highThresh)
    slice_BW = opening(slice_BW, disk(2))
    high_thresh_str = str(highThresh)
    kg = True
    while kg:
        temp = Imover(slice, slice_BW, [0.1, -0.2, -0.2])
        cv2.imshow('Threshold Image', temp)
        button = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if button == ord('y') or button == ord('Y'):
            kg = False
        else:
            high_thresh_str = input('Enter new {} threshold: '.format(name))
            slice_BW = np.logical_and(slice > lowThresh, slice < float(high_thresh_str))
            slice_BW = opening(slice_BW, disk(2))
    return float(high_thresh_str)

def DetermineThreshold(fftFiltered, deltaZ):
    fftFiltered = np.array(fftFiltered)

    smooth = gaussian(fftFiltered, sigma=0.5, mode='nearest', truncate=2.0)
    #Can also try
    #smooth = gaussian_filter(fftFiltered, sigma=0.5)

    smooth = zoom(smooth, zoom=(1,1,deltaZ))
    thresh = threshold_multiotsu(smooth, classes=4)
    
    thresh = ThreshFeedback(smooth, 0, thresh[0], 'pore')
    
    voids = smooth < thresh #Use > for light selection (EDS). Use < for dark selection (bubbles)

    return smooth, voids

def EDSThresholdCleaningMask(smooth, voids):
    dims = smooth.shape
    AoI = StackCropping(smooth)
    for i in tqdm(range(dims[2]), desc='Removing small pores and filling gaps'):
        temp = voids[:, :, i].copy()
        temp[~AoI] = 0
        voids[:, :, i] = temp
        voids[:, :, i] = opening(voids[:, :, i], disk(2))
        voids[:, :, i] = closing(voids[:, :, i], disk(5))
        voids[:, :, i] = cv2.fillPoly(np.uint8(voids[:, :, i]), [np.array(AoI, dtype=np.int32)], 255)

    # Optimizing pore contours for best fit
    for i in tqdm(range(dims[2]), desc='Optimizing pore contours for best fit'):
        voids[:, :, i] = active_contour(smooth[:, :, i], voids[:, :, i], max_iterations=20, boundary_condition='edge')

    return voids

def BubbleThresholdCleaningMask(smooth, voids):
    dims = smooth.shape
    AoI = StackCropping(smooth)

    for i in range(dims[2]):
        temp = ~cv2.threshold(smooth[:, :, i], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        temp = cv2.bitwise_not(temp)
        temp[~AoI] = 0
        voids[:, :, i] = temp

    voids = opening(voids, ball(1))
    voids = cv2.fillPoly(np.uint8(voids), [np.array(AoI, dtype=np.int32)], 255)
    voids = closing(voids, ball(3))

    for i in range(dims[2]):
        voids[:, :, i] = cv2.fillPoly(np.uint8(voids[:, :, i]), [np.array(AoI, dtype=np.int32)], 255)

    voids = active_contour(smooth, voids, max_iterations=100, boundary_condition='edge')

    for i in range(dims[2]):
        voids[:, :, i] = cv2.fillPoly(np.uint8(voids[:, :, i]), [np.array(AoI, dtype=np.int32)], 255)

    voids = opening(voids, ball(1))
    voids = voids.astype(bool)
    
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

def main():
    global inDir, outDir, saveImages
    # inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Unprocessed/Stack3_D"
    inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/ProcessedPython/Stack3_D/FFTFiltered"
    outDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/ProcessedPython/Stack3_D"
    saveImages = False

    # images = readImages(inDir)
    # rotated = StackRotate(images, tiltAngle=-3.5)
    # shifted = StackVerticalShift(rotated, shift=0.3497942386831276) #Enter + value for shift if shift margin is known
    # cropped = StackCropping(shifted)
    # filtered = NoiseReduction(cropped)
    # fftFiltered = FFT_Filtering(filtered)

    fftFiltered = readImages(inDir)
    smooth, voids = DetermineThreshold(fftFiltered, 59)

if __name__ == "__main__":
    main()
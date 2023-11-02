import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import os
import cv2
from scipy.signal import wiener
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from matplotlib.widgets import RectangleSelector
from PIL import Image
from scipy.ndimage import zoom, binary_fill_holes
from skimage.filters import gaussian, threshold_multiotsu
from skimage.morphology import opening, closing, disk, ball
from skimage.segmentation import morphological_chan_vese
from tqdm import tqdm
from mayavi import mlab

#Reads images from a given directory of images. Does not read any files that do not have the specified file extension
#Assumes image filenames are in the format: filename_#.fileExtension where # is the number of the image
def ReadImages(inDir, fileExtension):
    image_list = os.listdir(inDir)
    image_list = [file for file in image_list if file.endswith(fileExtension)]

    def ExtractImgNumber(filename):
        return int(filename.split('_')[1].split('.tif')[0])
    
    image_list.sort(key=ExtractImgNumber)
    
    images = []
    for image_file in image_list:
        image_path = os.path.join(inDir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    images = np.array(images)

    return images

#Rotates each image in the image stack about its center by a given angle
#Note: Positive tiltAngle corresponds to CCW, Negative tiltAngle corresponds to CW
def StackRotate(images, tiltAngle, saveImages=False):
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

#Finds the Y coordinate of a manually selected point on an image
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

#Vertically shifts each image in the image stack until they are aligned with the first image slice in the stack
#Note: This function will manually compute the shift margin interactively unless a known shift margin is specified
def StackVerticalShift(rotated, shift=None, saveImages=False):
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

#Finds the Xmin, Ymin, Xmax, and Ymax values of a manually selected rectangular area on an image
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

def StackCropping(shifted, saveImages=False):
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

def NoiseReduction(cropped, saveImages=False):
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

#Creates a boolean mask where the areas of two manually selected rectangles on an image are set as True and everything else is set to False
#This function is needed to mask high frequencies correlating to curtaining effects
def CreateDoubleRectangleMask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.bool_)

    print("Select 1st rectangular AOI")
    R1x1, R1y1, R1x2, R1y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Left Rectangle
   
    print("Select 2nd rectangular AOI")
    R2x1, R2y1, R2x2, R2y2 = GetRectangleCoords(image) #Select top-left and bottom-right corner of Right Rectangle

    # Rectangles should extand to outer edges, hard to do so with GetCoords() so this extends the correct cordinates for each rectangle
    if R1x1 != 0:
        R1x1 = 0
    
    if R2x2 != w - 1:
        R2x2 = w - 1
    
    mask[R1y1:R1y2+1, R1x1:R1x2+1] = True
    mask[R2y1:R2y2+1, R2x1:R2x2+1] = True

    print(type(mask))
    plt.imshow(mask, cmap='gray')
    plt.show()
    return mask

#Performs Fourier Fast Transform filtering in the frequency domain of each image in the image stack to remove curtaining effects
def FFT_Filtering(filtered, saveImages=False):
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

            if ans == "Y" or ans == "y":
                applyFilterAgain = True
                break
            elif ans == "N" or ans == 'n':
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

#Expands an image stack about the number of image slices in the stack by a factor of deltaZ 
def ExpandImageStack(images, deltaZ):
    print("Expanding image stack...")
    return zoom(images, zoom=(round(deltaZ), 1, 1), order=2)

#Creates a mask-based image overlay
def Imover(inputImage, mask, color=[255, 255, 255]):
    #Ensures formattedImage and mask are uint8 and binary data types respectively
    formattedImage = np.array(inputImage)
    mask = (mask != 0)

    try:
        if formattedImage.ndim == 2:
            #Input is grayscale. Initialize all output channels the same
            outRed = np.copy(formattedImage)
            outGreen = np.copy(formattedImage)
            outBlue = np.copy(formattedImage)
        elif formattedImage.ndim == 3:
            #Input is RGB truecolor
            outRed = np.copy(formattedImage[:,:,0])
            outGreen = np.copy(formattedImage[:,:,1])
            outBlue = np.copy(formattedImage[:,:,2])
    except:
        print('Invalid image entered for Imover. Only grayscale or RGB images are accepted.')
        raise

    outRed[mask] = color[0]
    outGreen[mask] = color[1]
    outBlue[mask] = color[2]

    out = np.stack((outRed, outGreen, outBlue), axis=-1)
    out = out.astype(np.uint8)

    return out


#Manually determines what threshold is appropriate given 4 threshold values
def ThreshFeedback(img, thresh):
    lowThresh = 0
    threshIndex = 0
    highThresh = thresh[threshIndex]
    dims = img.shape

    #indexing method is smooth[z,x,y]

    slice1 = img[round(dims[0]/5),:,:]
    slice2 = img[round(dims[0]*2/5),:,:]
    slice3 = img[round(dims[0]*3/5),:,:]
    slice4 = img[round(dims[0]*4/5),:,:]

    slice = np.vstack([np.hstack([slice1, slice2]), np.hstack([slice3, slice4])])
    slice_BW = (slice > lowThresh) & (slice < highThresh)
    slice_BW = opening(slice_BW, disk(2))

    while True:
        temp = Imover(slice, slice_BW, [100, 0, 0])

        plt.imshow(temp)
        plt.show()

        button = input('Press [Y] if threshold is acceptable. Otherwise, enter a new threshold value: ')

        if button == 'y' or button == 'Y':
            break
        else:
            try:
                highThresh = float(button)
                slice_BW = (slice > lowThresh) & (slice < highThresh)
                slice_BW = opening(slice_BW, disk(2))
                temp = []
            except:
                print("Invalid threshold. Continuing loop with current threshold value.")
    
    return float(highThresh)

#Expands image and returns binary array containing the voids in each image slice based on a manually determined threshold value
def Thresholding(fftFiltered, deltaZ):
    fftFiltered = np.array(fftFiltered)

    smooth = gaussian(fftFiltered, sigma=0.5, mode='nearest', truncate=2.0)
    smooth = (smooth*255).astype(np.uint8)

    smooth = ExpandImageStack(smooth, deltaZ)

    thresh = threshold_multiotsu(smooth, classes=5)

    displayThresh = ', '.join(map(str, thresh))
    print('Generated guideline threshold values are: {}'.format(displayThresh))

    thresh = ThreshFeedback(smooth, thresh)

    voids = smooth < thresh
            
    return smooth, voids

def AltThreshFeedback(img, originalImg, thresh):
    dims = img.shape

    slice1 = img[round(dims[0]/5),:,:]
    slice2 = img[round(dims[0]*2/5),:,:]
    slice3 = img[round(dims[0]*3/5),:,:]
    slice4 = img[round(dims[0]*4/5),:,:]
    slice = np.vstack([np.hstack([slice1, slice2]), np.hstack([slice3, slice4])])
    
    slice_BW = slice > thresh[0]
    slice_BW = opening(slice_BW, disk(2))

    #OriginalSlice used to display generated mask on original image, easier to see effects of the different thresholds.
    originalSlice1 = originalImg[round(dims[0]/5),:,:]
    originalSlice2 = originalImg[round(dims[0]*2/5),:,:]
    originalSlice3 = originalImg[round(dims[0]*3/5),:,:]
    originalSlice4 = originalImg[round(dims[0]*4/5),:,:]
    originalSlice = np.vstack([np.hstack([originalSlice1, originalSlice2]), np.hstack([originalSlice3, originalSlice4])])

    while True:
        temp = Imover(originalSlice, slice_BW, [100, 0, 0])

        plt.imshow(temp)
        plt.show()

        button = input('Press [Y] if threshold is acceptable. Otherwise, enter a new threshold value: ')

        if button == 'y' or button == 'Y':
            break
        else:
            try:
                highThresh = float(button)
                slice_BW = slice > highThresh
                slice_BW = opening(slice_BW, disk(2))
                temp = []
            except:
                print("Invalid threshold. Continuing loop with current threshold value.")
    
    return float(highThresh)

def AltThresholding(fftFiltered):
    fftFiltered = np.array(fftFiltered)

    smooth = gaussian(fftFiltered, sigma=2.0, mode='nearest', truncate=2.0)
    smooth = np.array((smooth*255)).astype(np.uint8)

    modifiedSmooth = []
    for img in fftFiltered:
        temp = 255 - img
        temp = cv2.subtract(temp, img)
        modifiedSmooth.append(temp)
    modifiedSmooth = np.array(modifiedSmooth)

    thresh = threshold_multiotsu(modifiedSmooth, classes=5)

    displayThresh = ', '.join(map(str, thresh))
    print('Generated guideline threshold values are: {}'.format(displayThresh))

    thresh = AltThreshFeedback(modifiedSmooth, smooth, thresh)

    voids = smooth < thresh

    return smooth, voids

#Algorithm to clean up the binary array containing pore locations. Removes small pores, fills gaps in pores, and performs a contour fit 
def EDSThresholdCleaningMask(smooth, voids):
    smooth = np.array(smooth)
    voids = np.array(voids)
    dims = smooth.shape

    for i in tqdm(range(dims[0]), desc='Removing small pores and filling gaps'):
        voids[i,:,:] = opening(voids[i,:,:], disk(2))
        voids[i,:,:] = closing(voids[i,:,:], disk(5))
        voids[i,:,:] = binary_fill_holes(voids[i,:,:])

    for i in tqdm(range(dims[0]), desc='Optimizing pore contours for best fit'):
        voids[i,:,:] = morphological_chan_vese(smooth[i,:,:], 20, voids[i,:,:])
    
    return voids

#Does not work at all. Needs significant rework.
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

    voids = morphological_chan_vese(smooth, voids, max_iterations=100, boundary_condition='edge')

    for i in range(dims[2]):
        voids[:, :, i] = cv2.fillPoly(np.uint8(voids[:, :, i]), [np.array(AoI, dtype=np.int32)], 255)

    voids = opening(voids, ball(1))
    voids = voids.astype(bool)

#Creates a very rudimentary 3D reconstruction of a binary 3D image stack using Mayavi
def BasicReconstruction(voids):
    #Assumes voids is binary:
    voids = np.array(voids).astype(float)

    print('Creating 3D display...')
    mlab.figure()

    mlab.contour3d(voids, colormap='binary')

    mlab.show()

def SaveImages(images, folderName, outDir):
    images = np.array(images)

    if not os.path.exists(outDir):
        print("Invalid Directory")
    else:
        folderPath = os.path.join(outDir, folderName)    
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
        for i, image in enumerate(tqdm(images, desc='Saving Images')):
            fileName = f"{folderName}_{i}.tif"
            img = Image.fromarray(image)
            img.save(os.path.join(folderPath, fileName))

def main():
    global inDir, outDir, saveImages
    # inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Unprocessed/Stack3_D"
    inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/ProcessedPython/Stack3_D/Filtered"
    outDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/ProcessedPython/Stack3_D"
    saveImages = False

    depth = 59/4.56
    threshStyle = 'EDS'

    images = ReadImages(inDir, ".tif")
    rotated = StackRotate(images, tiltAngle=-3.5)
    shifted = StackVerticalShift(rotated, shift=0.3497942386831276) #Enter + value for shift if shift margin is known
    cropped = StackCropping(shifted)
    filtered = NoiseReduction(cropped)
    fftFiltered = FFT_Filtering(filtered)
    smooth, voids = Thresholding(fftFiltered, depth)

    if threshStyle == 'EDS':
        voids = EDSThresholdCleaningMask(smooth, voids)
    elif threshStyle == 'bubbles':
        voids = BubbleThresholdCleaningMask(smooth, voids)


if __name__ == "__main__":
    main()
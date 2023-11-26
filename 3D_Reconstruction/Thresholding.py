##### This code was used to test and determine appropriate threshold technique #####
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from scipy.ndimage import zoom, binary_fill_holes
from skimage.filters import gaussian, threshold_multiotsu, threshold_sauvola
from skimage.morphology import binary_opening, binary_closing, binary_dilation, disk
from skimage.segmentation import morphological_chan_vese
from tqdm import tqdm
from mayavi import mlab
import time
from stl import mesh


#Assumes image filenames are in the format: filename_#.fileExtension where # is the number of the image
def ReadImages(inDir, fileExtension):
    image_list = os.listdir(inDir)
    image_list = [file for file in image_list if file.endswith(fileExtension)]
    image_list.sort()
    
    images = []
    for image_file in image_list:
        image_path = os.path.join(inDir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    images = np.array(images)

    return images

def ExpandImageStack(images, deltaZ):
    print("Expanding image stack...")
    return zoom(images, zoom=(round(deltaZ), 1, 1), order=2)

def CreateComboImg(imgStack):
    dims1 = imgStack.shape
    slice1 = imgStack[round(dims1[0]/5)]
    slice2 = imgStack[round(dims1[0]*2/5)]
    slice3 = imgStack[round(dims1[0]*3/5)]
    slice4 = imgStack[round(dims1[0]*4/5)]

    ComboImg = np.vstack([np.hstack([slice1, slice2]), np.hstack([slice3, slice4])])

    return ComboImg

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
            outRed = np.copy(formattedImage[0])
            outGreen = np.copy(formattedImage[1])
            outBlue = np.copy(formattedImage[2])
    except:
        print('Invalid image entered for Imover. Only grayscale or RGB images are accepted.')
        raise

    outRed[mask] = color[0]
    outGreen[mask] = color[1]
    outBlue[mask] = color[2]

    out = np.stack((outRed, outGreen, outBlue), axis=-1)
    out = out.astype(np.uint8)

    return out

def ThreshFeedback(img, thresh):
    lowThresh = 0
    highThresh = thresh[0]

    slice = CreateComboImg(img)

    slice_BW = (slice > lowThresh) & (slice < highThresh)
    slice_BW = binary_opening(slice_BW, disk(2))

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
                slice_BW = binary_opening(slice_BW, disk(2))
                temp = []
            except:
                print("Invalid threshold. Continuing loop with current threshold value.")
    
    return float(highThresh)

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
    slice = CreateComboImg(img)
    
    slice_BW = slice > thresh[0]
    slice_BW = binary_opening(slice_BW, disk(2))

    #OriginalSlice used to display generated mask on original image, easier to see effects of the different thresholds.
    originalSlice = CreateComboImg(originalImg)

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
                slice_BW = binary_opening(slice_BW, disk(2))
                temp = []
            except:
                print("Invalid threshold. Continuing loop with current threshold value.")
    
    return float(highThresh)

def AltThresholding(fftFiltered, deltaZ):
    fftFiltered = np.array(fftFiltered)

    smooth = gaussian(fftFiltered, sigma=0.5, mode='nearest', truncate=2.0)
    smooth = np.array((smooth*255)).astype(np.uint8)

    smooth = ExpandImageStack(smooth, deltaZ)

    modifiedSmooth = []
    for img in smooth:
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

def ThresholdCleaningMask(smooth, voids, imageStack = True):
    smooth = np.array(smooth)
    voids = np.array(voids)
    dims = smooth.shape
    
    if imageStack == True:
        for i in tqdm(range(dims[0]), desc='Optimizing pore contours for best fit'):
            voids[i] = binary_opening(voids[i], disk(1))
            voids[i] = binary_fill_holes(voids[i])
            voids[i] = binary_closing(voids[i], disk(3))
            voids[i] = binary_dilation(voids[i], disk(1))
            voids[i] = morphological_chan_vese(smooth[i], 20, voids[i])
        voids = binary_fill_holes(voids)
    else: #Single image given, for threshold testing purposes
        voids = binary_opening(voids, disk(1))
        voids = binary_fill_holes(voids)
        voids = binary_closing(voids, disk(3))
        voids = binary_dilation(voids, disk(1))
        voids = morphological_chan_vese(smooth, 20, voids)
        voids = binary_fill_holes(voids)

    return voids

def EnhanceContrast(img, lowerPercentile, upperPercentile):
    img = np.array(img)
    lowerLimit, upperLimit = np.percentile(img, [lowerPercentile * 100, upperPercentile * 100])
    stretchedImg = np.clip((img - lowerLimit) / (upperLimit - lowerLimit), 0, 1)
    stretchedImg = np.array(stretchedImg*255).astype(np.uint8)

    return stretchedImg

def SauvolaThresholding(fftFiltered, deltaZ, contrastLowerPercentile, contrastUpperPercentile, sigma, threshSize):
    smooth = np.copy(fftFiltered)
    smooth = ExpandImageStack(smooth, deltaZ)

    voids = []
    for i, img in enumerate(tqdm(smooth, desc='Applying Sauvola Image Threshold')):
        originalImg = img
        img = EnhanceContrast(img, contrastLowerPercentile, contrastUpperPercentile)
        img = gaussian(img, sigma=sigma, mode='nearest', truncate=2.0)
        img = np.array(img*255).astype(np.uint8)
        thresh = threshold_sauvola(img, window_size=threshSize)
        void = originalImg > thresh
        voids.append(void)
    
    return smooth, np.array(voids)

def EdgeDetectionThresholding(fftFiltered, deltaZ, contrastLowerPercentile, contrastUpperPercentile, sigma, thresh1, thresh2):
    smooth = np.copy(fftFiltered)
    smooth = ExpandImageStack(smooth, deltaZ)

    voids = []
    for i, img in enumerate(tqdm(smooth, desc='Applying Edge Detection Image Threshold')):
        originalImg = img
        temp = EnhanceContrast(img, contrastLowerPercentile, contrastUpperPercentile)
        temp = gaussian(temp, sigma=sigma, mode='nearest', truncate=2.0)
        temp = np.array(temp*255).astype(np.uint8)
        void = cv2.Canny(temp, thresh1, thresh2)
        void = binary_dilation(void, disk(1))
        void = morphological_chan_vese(originalImg, 20, void)
        voids.append(void)

    return np.array(voids)

def CreateOverlayComboImg(imgStack, voids):
    dims = imgStack.shape
    
    slice1 = imgStack[round(dims[0]/5)]
    slice2 = imgStack[round(dims[0]*2/5)]
    slice3 = imgStack[round(dims[0]*3/5)]
    slice4 = imgStack[round(dims[0]*4/5)]
    
    voids1 = voids[round(dims[0]/5)]
    voids2 = voids[round(dims[0]*2/5)]
    voids3 = voids[round(dims[0]*3/5)]
    voids4 = voids[round(dims[0]*4/5)]

    slice1 = Imover(slice1, voids1, color=[100,0,0])
    slice2 = Imover(slice2, voids2, color=[100,0,0])
    slice3 = Imover(slice3, voids3, color=[100,0,0])
    slice4 = Imover(slice4, voids4, color=[100,0,0])

    ComboImg = np.vstack([np.hstack([slice1, slice2]), np.hstack([slice3, slice4])])

    return ComboImg

def CompareImages(ImgStack1, Voids1, Img1Name, ImgStack2, Voids2, Img2Name, figsize=[10,5]):
    Img1 = CreateOverlayComboImg(ImgStack1, Voids1)
    Img2 = CreateOverlayComboImg(ImgStack2, Voids2)
    
    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.imshow(Img1, cmap='gray')
    plt.title(f'{Img1Name}')

    plt.subplot(1,2,2)
    plt.imshow(Img2, cmap='gray')
    plt.title(f'{Img2Name}')

    plt.show()

def BasicReconstruction(voids):
    #Assumes voids is binary:
    voids = np.array(voids).astype(float)

    print('Creating 3D display...')
    mlab.figure()

    mlab.contour3d(voids, colormap='binary')

    mlab.show()

def CreateMeshReconstruction(voids):
    voids[0,:,:] = False
    voids[-1,:,:] = False
    
    verts, faces, normals, values = measure.marching_cubes(voids)

    obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, f in enumerate(faces):
        obj.vectors[i] = verts[f]
    
    return obj

def SaveImages(images, folderName, outDir):
    images = np.array(images)

    if not os.path.exists(outDir):
        raise ValueError(f'Invalid directory. Directory {outDir} does not exist.')

    else:
        folderPath = os.path.join(outDir, folderName)    
        
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        
        numDigits = len(str(len(images)))

        for i, image in enumerate(tqdm(images, desc='Saving Images')):
            fileName = f'{folderName}{i:0{numDigits}}.tif'
            img = Image.fromarray(image)
            img.save(os.path.join(folderPath, fileName))

def SaveMeshAsSTL(mesh3D, fileName, folderName, outDir):
    if not os.path.exists(outDir):
        raise ValueError(f'Invalid directory. Directory {outDir} does not exist.')
    else:
        folderPath = os.path.join(outDir, folderName)

        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        fileName = f'{fileName}.stl'
        mesh3D.save(os.path.join(folderPath, fileName))

def main():
    # inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/Unprocessed/Stack3_D"
    inDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/ProcessedPython/Stack3_D/FFTFiltered"
    outDir = "C:/Users/Cade Finney/Desktop/Research/PoreCharacterizationFiles/ProcessedPython/Stack3_D"

    depth = 59/4.56

    ##### .stl creation code #####
    testBatchSize = 1

    fftFiltered = ReadImages(inDir, '.tif')

    batchSize = round(len(fftFiltered)*testBatchSize)
    batch = np.copy(fftFiltered[:batchSize])

    voids = EdgeDetectionThresholding(batch, depth, 0.01, 0.99, 2.0, 50, 100)

    voids[0,:,:] = False
    voids[-1,:,:] = False
    
    verts, faces, normals, values = measure.marching_cubes(voids)

    obj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, f in enumerate(faces):
        obj.vectors[i] = verts[f]
    
    SaveMeshAsSTL(obj, 'PoreReconstruction', '3D_Reconstruction', outDir)

    ##### Threshold Comparison Code #####
    # testBatchSize = 0.05

    # fftFiltered = ReadImages(inDir, '.tif')

    # batchSize = round(len(fftFiltered)*testBatchSize)
    # batch = np.copy(fftFiltered[:batchSize])

    # expandedBatch = ExpandImageStack(batch, depth)
    
    # start1 = time.time()
    # smooth1, voids1 = Thresholding(batch, depth)
    # voids1 = ThresholdCleaningMask(smooth1, voids1)
    # end1 = time.time()
    # time1 = end1 - start1

    # start2 = time.time()
    # smooth2, voids2 = AltThresholding(batch, depth)
    # voids2 = ThresholdCleaningMask(smooth2, voids2)
    # end2 = time.time()
    # time2 = end2 - start2

    # start3 = time.time()
    # smooth3, voids3 = SauvolaThresholding(batch, depth, 0.01, 0.99, 2.0, 25)
    # voids3 = ThresholdCleaningMask(smooth3, voids3)
    # end3 = time.time()
    # time3 = end3 - start3

    # start4 = time.time()
    # voids4 = EdgeDetectionThresholding(batch, depth, 0.01, 0.99, 2.0, 50, 100)
    # end4 = time.time()
    # time4 = end4 - start4
    
    # print(f'Original Threshold took {time1} seconds')
    # print(f'Alternate Threshold took {time2} seconds')
    # print(f'Sauvola Threshold took {time3} seconds')
    # print(f'Edge Detection Threshold took {time4} seconds')

    # CompareImages(expandedBatch, voids1, 'Original Threshold', expandedBatch, voids2, 'Alternate Threshold')
    # CompareImages(expandedBatch, voids1, 'Original Threshold', expandedBatch, voids3, 'Sauvola Threshold')
    # CompareImages(expandedBatch, voids1, 'Original Threshold', expandedBatch, voids4, 'Edge Detection Threshold')    

if __name__ == '__main__':
    main()
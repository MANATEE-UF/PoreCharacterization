import numpy as np
import scipy.special
import scipy.stats
from skimage import io
from skimage.color import gray2rgb
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import os

# Class used to aid in displaying the image with grid overlayed onto sampled pixels
class PixelMap:
    def __init__(self,image:np.ndarray):
        self.rows = len(image)
        self.cols = len(image[0])
        self.numPixels = self.rows * self.cols
        self.originalImage = image # grayscale

        if np.max(image) <= 1.0:
            self.originalImage *= 255
            self.originalImage = self.originalImage.astype(int)

    # Overlay a grid onto the original image and return centered around that grid
    def GetImageWithGridOverlay(self, pixelRow:int, pixelCol:int, newColor:tuple, numSurroundingPixels:int) -> np.ndarray: 
        # Center
        displayImage = gray2rgb(self.originalImage)

        displayImage[pixelRow][pixelCol] = newColor

        # Three above
        maxVal = 3 if pixelRow > 2 else pixelRow
        for i in range(1,maxVal):
            displayImage[pixelRow-i][pixelCol] = newColor

        # Three below
        maxVal = 3 if pixelRow < self.rows-3 else self.rows-pixelRow
        for i in range(1,maxVal):
            displayImage[pixelRow+i][pixelCol] = newColor

        # Three right
        maxVal = 3 if pixelCol < self.cols-3 else self.cols-pixelCol
        for i in range(1,maxVal):
            displayImage[pixelRow][pixelCol+i] = newColor

        # Three left
        maxVal = 3 if pixelCol > 2 else pixelCol
        for i in range(1,maxVal):
            displayImage[pixelRow][pixelCol-i] = newColor
        
        # pad image to ensure display proper
        displayImage = np.pad(displayImage, ((numSurroundingPixels+1, numSurroundingPixels+1), (numSurroundingPixels+1, numSurroundingPixels+1), (0,0)))

        # Crop the image to center around the grid with numSurroundingPixels around
        displayImage = displayImage[pixelRow:pixelRow+2*numSurroundingPixels,pixelCol:pixelCol+2*numSurroundingPixels,:]

        return displayImage

    def GetCroppedImage(self, leftBound, rightBound, topBound, bottomBound):
        return self.originalImage[topBound:bottomBound, leftBound:rightBound]


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class InitialGuessWidget(QtWidgets.QWidget):
    
    def __init__(self, parentTab, imagePath, numStrata_N):
        super(InitialGuessWidget,self).__init__()

        self.imagePath = imagePath
        self.numStrata_N = numStrata_N
        self.strataIndex = 0
        self.initialGuesses = []

        self.parentTab = parentTab
        
        # 5, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 95
        self.fivePctButton = QtWidgets.QPushButton("5%")
        self.fivePctButton.clicked.connect(lambda: self.LogEstimate(0.05))
        self.twelvePctButton = QtWidgets.QPushButton("12.5%")
        self.twelvePctButton.clicked.connect(lambda: self.LogEstimate(0.125))
        self.twntyFivePctButton = QtWidgets.QPushButton("25%")
        self.twntyFivePctButton.clicked.connect(lambda: self.LogEstimate(0.25))
        self.thirtySevenPctButton = QtWidgets.QPushButton("37.5%")
        self.thirtySevenPctButton.clicked.connect(lambda: self.LogEstimate(0.375))
        self.fiftyPctButton = QtWidgets.QPushButton("50%")
        self.fiftyPctButton.clicked.connect(lambda: self.LogEstimate(0.50))
        self.sixtyTwoPctButton = QtWidgets.QPushButton("62.5%")
        self.sixtyTwoPctButton.clicked.connect(lambda: self.LogEstimate(0.625))
        self.seventyFivePctButton = QtWidgets.QPushButton("75%")
        self.seventyFivePctButton.clicked.connect(lambda: self.LogEstimate(0.75))
        self.eightySevenPctButton = QtWidgets.QPushButton("87.5%")
        self.eightySevenPctButton.clicked.connect(lambda: self.LogEstimate(0.875))
        self.ninetyFivePctButton = QtWidgets.QPushButton("95%")
        self.ninetyFivePctButton.clicked.connect(lambda: self.LogEstimate(0.95))

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.fivePctButton)
        hbox.addWidget(self.twelvePctButton)
        hbox.addWidget(self.twntyFivePctButton)
        hbox.addWidget(self.thirtySevenPctButton)
        hbox.addWidget(self.fiftyPctButton)
        hbox.addWidget(self.sixtyTwoPctButton)
        hbox.addWidget(self.seventyFivePctButton)
        hbox.addWidget(self.eightySevenPctButton)
        hbox.addWidget(self.ninetyFivePctButton)
        
        self.buttonRegion = QtWidgets.QWidget()
        self.buttonRegion.setLayout(hbox)

        vbox = QtWidgets.QVBoxLayout()

        self.sc = MplCanvas(self, width=7, height=7, dpi=100)
        vbox.addWidget(self.sc)
        vbox.addWidget(self.buttonRegion)
        self.setLayout(vbox)

        self.originalImage = io.imread(imagePath, as_gray=True)

        self.myMap = PixelMap(self.originalImage)
        self.N = self.myMap.numPixels

        self.leftBounds = []
        self.rightBounds = []
        self.topBounds = []
        self.bottomBounds = []

        for i in range(self.numStrata_N):
            topBound = int(i*self.myMap.rows/self.numStrata_N)
            bottomBound = int((i+1)*self.myMap.rows/self.numStrata_N)
            for j in range(self.numStrata_N):
                leftBound = int(j*self.myMap.cols/self.numStrata_N)
                rightBound = int((j+1)*self.myMap.cols/self.numStrata_N)

                self.leftBounds.append(leftBound)
                self.rightBounds.append(rightBound)
                self.topBounds.append(topBound)
                self.bottomBounds.append(bottomBound)

        self.sc.axes.cla()
        self.sc.axes.imshow(self.myMap.GetCroppedImage(self.leftBounds[self.strataIndex], 
                                                       self.rightBounds[self.strataIndex], 
                                                       self.topBounds[self.strataIndex], 
                                                       self.bottomBounds[self.strataIndex]),
                                                       cmap="gray", vmin=0, vmax=255)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()

    def LogEstimate(self, value):
        self.initialGuesses.append(value)

        self.strataIndex += 1

        if self.strataIndex == self.numStrata_N**2:
            print("Strata estimates:")
            print(self.initialGuesses)
            self.parentTab.MoveToNextWidget(np.array(self.initialGuesses))
            return

        self.sc.axes.cla()
        self.sc.axes.imshow(self.myMap.GetCroppedImage(self.leftBounds[self.strataIndex], 
                                                       self.rightBounds[self.strataIndex], 
                                                       self.topBounds[self.strataIndex], 
                                                       self.bottomBounds[self.strataIndex]),
                                                       cmap="gray", vmin=0, vmax=255)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()


class PoreAnalysisWidget(QtWidgets.QWidget):

    def __init__(self, parentTab, saveDir, imagePath, numStrata_N, alpha, MOE, e_moe, widgetIndex, useOptimalAlloc=True):
        super(PoreAnalysisWidget, self).__init__()

        self.parentTab = parentTab
        self.widgetIndex = widgetIndex

        self.useOptimalAlloc = useOptimalAlloc

        self.saveDir = saveDir

        self.sc = MplCanvas(self, width=7, height=7, dpi=100)

        vbox = QtWidgets.QVBoxLayout()

        hbox2 = QtWidgets.QHBoxLayout()

        self.lastEntryText = QtWidgets.QLabel("Last Entry: --")
        self.lastEntryText.setAlignment(QtCore.Qt.AlignCenter)
        self.lastEntryText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        self.indexProgressText = QtWidgets.QLabel(f"Sample: -/-, Strata: -/-")
        self.indexProgressText.setAlignment(QtCore.Qt.AlignCenter)
        self.indexProgressText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        hbox2.addWidget(self.lastEntryText)
        hbox2.addWidget(self.indexProgressText)
        
        self.zoomOutButton = QtWidgets.QPushButton("Zoom Out")
        self.zoomOutButton.clicked.connect(self.ZoomOut)
        self.zoomInButton = QtWidgets.QPushButton("Zoom In")
        self.zoomInButton.clicked.connect(self.ZoomIn)

        hbox2.addWidget(self.zoomOutButton)
        hbox2.addWidget(self.zoomInButton)

        vbox.addLayout(hbox2)
        vbox.addWidget(self.sc)
        hbox = QtWidgets.QHBoxLayout()

        leftText = QtWidgets.QLabel("Left Arrow Key For 0")
        leftText.setAlignment (QtCore.Qt.AlignCenter)
        leftText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        upText = QtWidgets.QLabel("Up Arrow Key For 0.5")
        upText.setAlignment (QtCore.Qt.AlignCenter)
        upText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        rightText = QtWidgets.QLabel("Right Arrow Key For 1")
        rightText.setAlignment (QtCore.Qt.AlignCenter)
        rightText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        downText = QtWidgets.QLabel("Down Arrow Key To Go Back")
        downText.setAlignment (QtCore.Qt.AlignCenter)
        downText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(upText)
        vbox2.addWidget(downText)

        hbox.addWidget(leftText)
        hbox.addLayout(vbox2)
        hbox.addWidget(rightText)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.imageName = imagePath
        image = io.imread(imagePath, as_gray=True)
        self.myMap = PixelMap(image)
        self.N = self.myMap.numPixels
        
        self.numSurroundingPixels = 50

        self.alpha = alpha
        self.MOE = MOE
        self.e_moe = e_moe
        self.d = 0.5
        self.numStrata_N = numStrata_N
        self.N_h = int(self.N / self.numStrata_N**2)

    def InitializeCounting(self, initialGuesses):

        # ############################################################################ #
        # Calculate the total number of samples needed to acheieve specified precision #
        # ############################################################################ #

        W_h = 1 / self.numStrata_N**2 # Value constant because image is split evenly. small differences neglected
        initialStrataProportion = np.sum(initialGuesses) * self.N_h / self.N
        highVarGuess = np.ones(int(self.numStrata_N**2)) * 0.5
        variance = np.sum(W_h * np.sqrt(highVarGuess * (1 - highVarGuess)))**2 / self.numStrata_N**2 - ((1/self.N) * np.sum(W_h * highVarGuess * (1 - highVarGuess))) # highest variance given the initial guesses, assuming only one point taken per stratum
        if initialStrataProportion > 0.5 and self.MOE > 1-initialStrataProportion:
            self.MOE = ((1-initialStrataProportion)+self.MOE) / 2 # want to keep +- MOE as close as possible on the open side. so if p=0.01, with 5% MOE, the CI should be (0,0.06)
            # print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {MOE:.2f}")
        elif initialStrataProportion < 0.5 and self.MOE > initialStrataProportion:
            self.MOE = (initialStrataProportion + self.MOE) / 2
            # print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {initialStrataProportion:.2f}")
        upperCL = 1.0
        lowerCL = 0.0
        withinTolerance = False
        d = self.d
        currentIter = 0
        maxIters = 100
        currentIter = 0
        while not withinTolerance and currentIter < maxIters:
            n = np.sum(W_h * np.sqrt(initialGuesses * (1-initialGuesses)) / variance) # smaller variance, higher n
            n = int(np.ceil(n))

            lowerCL, upperCL = self.TwoSidedCL_A(n, initialStrataProportion, self.alpha)
            lowerCL /= n
            upperCL /= n

            # FIXME: Assumes that variance will always be too high to start. sometimes it starts too low, then d gets too small over time and eventually maxiters is reached
            if ((upperCL - lowerCL) / 2) > self.MOE: # Eq.15 not satisfied
                variance *= d
            else: # Eq. 15 satisfied
                pctDiff = abs((((upperCL - lowerCL) / 2) - self.MOE) / self.MOE)
                if pctDiff > self.e_moe: # variance too low, overestimating how many sample points needed
                    variance /= d
                    d += (1-d)*0.2
                    variance *= d
                else:
                    withinTolerance = True
            currentIter += 1

        # ###################################################### #
        # Allocate the total number of samples across the strata #
        # ###################################################### #

        def OptimalAllocation(n):
            strataVars = np.empty(self.numStrata_N**2)
            cnt = 0
            for i in range(self.numStrata_N):
                topBound = int(i*self.myMap.rows/self.numStrata_N)
                bottomBound = int((i+1)*self.myMap.rows/self.numStrata_N)
                for j in range(self.numStrata_N):
                    leftBound = int(j*self.myMap.cols/self.numStrata_N)
                    rightBound = int((j+1)*self.myMap.cols/self.numStrata_N)
                    strataVars[cnt] = np.var(self.myMap.originalImage[topBound:bottomBound,leftBound:rightBound])
                    cnt += 1

            n_h = np.empty(self.numStrata_N**2)

            n_h = np.ceil(n * self.N_h * strataVars / np.sum(self.N_h * strataVars)).astype(int)

            # Ensure that at least one point sampled per strata
            n_h = np.maximum(n_h, 1)

            return n_h

        def ProportionalAllocation(n):
            n_h = np.ones(self.numStrata_N**2) * (n/self.numStrata_N**2)

            n_h = np.ceil(n_h).astype(int)

            return n_h

        if self.useOptimalAlloc:
            n_h = OptimalAllocation(n)
        else:
            n_h = ProportionalAllocation(n)
        

        print(f"{np.sum(n_h)} samples needed to achieve {self.MOE} MOE with {100*(self.alpha)}% CI")

        print(n_h)

        self.n_h = n_h

        # ########################## #
        # Get pixel sample locations #
        # ########################## #

        pixels = []
        cnt = 0
        for i in range(self.numStrata_N):
            topBound = int(i*self.myMap.rows/self.numStrata_N)
            bottomBound = int((i+1)*self.myMap.rows/self.numStrata_N)
            for j in range(self.numStrata_N):
                leftBound = int(j*self.myMap.cols/self.numStrata_N)
                rightBound = int((j+1)*self.myMap.cols/self.numStrata_N)

                # TODO: Is below still a bug?
                # FIXME: Because separating out x and y selection, sometimes n_h is greater than num pixels in x or y direction, thus ValueError
                # randomY = np.random.choice(np.arange(topBound, bottomBound), n_h[cnt], replace=False) 
                # randomX = np.random.choice(np.arange(leftBound, rightBound), n_h[cnt], replace=False)

                random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h[cnt], replace=False)
                random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                random[0,:] += topBound
                random[1,:] += leftBound
        
                pixels.append(list(zip(random[0,:], random[1,:])))

                cnt += 1
        
        self.samplePositions = pixels # First axis is strata axis, second axis is sample axis
        self.numGrids = np.sum(n_h)

        # ############# #
        # Begin display #
        # ############# #

        self.sampleIndex = 0
        self.strataIndex = 0
        self.gridIndex = 0 # Used to track flattend index, useful for writing to 1D poreData

        self.poreData = np.zeros((self.numGrids))
        self.indexProgressText.setText(f"Sample: {self.sampleIndex+1}/{self.n_h[self.strataIndex]}, Strata: {self.strataIndex+1}/{self.numStrata_N**2}")

        displayImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(displayImage, cmap="gray", vmin=0, vmax=255)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()

        self.setFocus(QtCore.Qt.NoFocusReason) # Needed or the keyboard will not work

    def TwoSidedCL_A(self, n, p, alpha):
        sum = 0
        A_U = 0
        while sum <= alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
            A_U += 1
            sum = 0
            for i in range(A_U):
                sum += scipy.stats.binom.pmf(i, n, p)

        sum = 1.0
        A_L = 0
        while sum > alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
            A_L += 1
            sum = 0
            for i in range(A_L, n+1):
                sum += scipy.stats.binom.pmf(i, n, p)
        
        return A_L-1, A_U-1

    def ZoomOut(self):
        if self.numSurroundingPixels < 300:
            self.numSurroundingPixels += 25
        
        newImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()

        if self.numSurroundingPixels >= 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)

        if self.numSurroundingPixels <= 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)
        
        self.setFocus(QtCore.Qt.NoFocusReason) # Needed or the keyboard will not work
    
    def ZoomIn(self):
        if self.numSurroundingPixels > 25:
            self.numSurroundingPixels -= 25
        
        newImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], (50, 225, 248), self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()

        if self.numSurroundingPixels == 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)

        if self.numSurroundingPixels == 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)
        
        self.setFocus(QtCore.Qt.NoFocusReason) # Needed or the keyboard will not work
        
    def keyPressEvent(self, event):
        if self.parentTab.stackedWidget.currentIndex() != self.widgetIndex:
            return
        
        if event.key() == QtCore.Qt.Key_Left:
            self.RecordDataPoint(0)
            self.lastEntryText.setText("Last Data Entry: 0")
        elif event.key() == QtCore.Qt.Key_Up:
            self.RecordDataPoint(0.5)
            self.lastEntryText.setText("Last Data Entry: 0.5")
        elif event.key() == QtCore.Qt.Key_Right:
            self.RecordDataPoint(1)
            self.lastEntryText.setText("Last Data Entry: 1")
        elif event.key() == QtCore.Qt.Key_Down:
            self.RecordDataPoint(-1)
            self.lastEntryText.setText("Last Data Entry: Back")
        elif event.key() == QtCore.Qt.Key_Plus:
            self.ZoomIn()
        elif event.key() == QtCore.Qt.Key_Minus:
            self.ZoomOut()
        else:
            pass
    
    def RecordDataPoint(self, value):
        # -1 is go back
        if value == -1 and self.sampleIndex > 0: # move back one sample
            self.sampleIndex -= 1
            self.gridIndex -= 1
        elif value == -1 and self.sampleIndex == 0 and self.strataIndex > 0: # move to end of last strata
            self.strataIndex -= 1
            self.sampleIndex = self.n_h[self.strataIndex] - 1
            self.gridIndex -= 1
        elif value == -1 and self.sampleIndex == 0 and self.strataIndex == 0: # at the beginning of samples, do nothing
            pass
        else: # record value
            self.poreData[self.gridIndex] = value
            self.gridIndex += 1

            # At last grid index, move to next pore index
            if self.sampleIndex == self.n_h[self.strataIndex] - 1:
                self.sampleIndex = 0
                self.strataIndex += 1           
            else:
                self.sampleIndex += 1

        if self.gridIndex >= self.numGrids:
            p_h = []
            if self.saveDir is None:
                saveFileName = f"{os.path.splitext(os.path.basename(self.imageName))[0]}_countData.csv"
            else:
                saveFileName = f"{self.saveDir}/{os.path.splitext(os.path.basename(self.imageName))[0]}_countData.csv"
            with open(saveFileName, "a") as f:
                for i, n in enumerate(self.n_h):
                    if i == 0:
                        bounds = [0, n]
                    else:
                        bounds = [np.cumsum(self.n_h[:i])[-1], np.cumsum(self.n_h[:i])[-1] + n]
                    p_h.append(np.average(self.poreData[bounds[0]:bounds[1]]))

                p_st = np.sum(p_h) * self.N_h / self.N

                p_h = np.array(p_h)
                variance = (1 / self.N)**2 * np.sum((self.N_h**2 * (self.N_h-self.n_h) * p_h * (1-p_h)) / ((self.N_h-1) * self.n_h))

                lowerCL, upperCL = self.TwoSidedCL_A(np.sum(self.n_h), p_st, self.alpha)
                lowerCL /= np.sum(self.n_h)
                upperCL /= np.sum(self.n_h)

                np.savetxt(f, 
                    p_h, 
                    fmt="%.2f", 
                    delimiter=",", 
                    header=f"{os.path.splitext(os.path.basename(self.imageName))[0]} with {self.numGrids} samples across {self.numStrata_N**2} strata. \n alpha = {self.alpha} \n MOE = {self.MOE} \n e_moe = {self.e_moe} \n d = {self.d}",
                    footer=f"\n Porosity: {p_st * 100}% \n {self.alpha}% CI: ({100*lowerCL, 100*upperCL}) \n MOE: {(upperCL - lowerCL) / 2}")
            
            if self.useOptimalAlloc:
                print("Optimal Allocation:")
            else:
                print("Proportional Allocation:")
            print(f"\n Porosity: {p_st * 100:.3f}% \n {100*(self.alpha)}% CI: ({100*lowerCL:.3f}, {100*upperCL:.3f}) \n MOE: {(upperCL - lowerCL) / 2:.3f} \n Variance: {variance:.10f}")
            print()

            if self.useOptimalAlloc:
                self.parentTab.optAlloc_p_st = p_h
                self.parentTab.optAlloc_all = self.poreData
            else:
                self.parentTab.propAlloc_p_st = p_h
                self.parentTab.propAlloc_all = self.poreData
            
            self.parentTab.MoveToNextWidget()

            return

        markerColor = (50, 225, 248)
        newImage = self.myMap.GetImageWithGridOverlay(self.samplePositions[self.strataIndex][self.sampleIndex][0], self.samplePositions[self.strataIndex][self.sampleIndex][1], markerColor, self.numSurroundingPixels)
        self.indexProgressText.setText(f"Sample: {self.sampleIndex+1}/{self.n_h[self.strataIndex]}, Strata: {self.strataIndex+1}/{self.numStrata_N**2}")

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage, cmap="gray", vmin=0, vmax=255)
        self.sc.axes.set_yticks([])
        self.sc.axes.set_xticks([])
        self.sc.draw()


class MyWindow(QMainWindow):

    def __init__(self, imagePath, numStrata_N, alpha, MOE, e_moe, saveDir=None, useBothAllocations = False):
        super(MyWindow,self).__init__()

        self.saveDir = saveDir
        self.useBothAllocations = useBothAllocations
        self.initialGuesses = None

        self.optAlloc_p_st = []
        self.optAlloc_all = []
        self.propAlloc_p_st = []
        self.propAlloc_all = []

        self.setWindowTitle("SEMPLE")

        self.stackedWidget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stackedWidget)

        self.widget1 = InitialGuessWidget(self, imagePath, numStrata_N)
        self.widget2 = PoreAnalysisWidget(self, saveDir, imagePath, numStrata_N, alpha, MOE, e_moe, widgetIndex=1)

        self.stackedWidget.addWidget(self.widget1)
        self.stackedWidget.addWidget(self.widget2)

        if self.useBothAllocations:
            self.widget3 = PoreAnalysisWidget(self, saveDir, imagePath, numStrata_N, alpha, MOE, e_moe, widgetIndex=2, useOptimalAlloc=False)
            self.stackedWidget.addWidget(self.widget3)
        
        self.stackedWidget.setCurrentIndex(0)
        
        self.show()

        self.stackedWidget.setFocus(QtCore.Qt.NoFocusReason)

    def MoveToNextWidget(self, initialGuesses=None):
        if initialGuesses is not None:
            self.initialGuesses = initialGuesses

        # Update widget index
        if self.stackedWidget.currentIndex()+1 == self.stackedWidget.count():
            stratTest = scipy.stats.ttest_ind(self.optAlloc_p_st, self.propAlloc_p_st)
            allTest = scipy.stats.ttest_ind(self.optAlloc_all, self.propAlloc_all)

            print(stratTest.statistic)
            print(stratTest.pvalue)

            print(allTest.statistic)
            print(allTest.pvalue)
            quit()
        else:
            self.stackedWidget.setCurrentIndex(self.stackedWidget.currentIndex() + 1)
        
        self.stackedWidget.currentWidget().InitializeCounting(self.initialGuesses)

# TODO: Confirmation on last grid before submitting
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    files = os.listdir("OptimalAllocationStudy")
    files.sort()
    filename = f"OptimalAllocationStudy/{files[0]}"
    print(filename)
    alpha = 0.95
    MOE = 0.05
    e_moe=0.01
    strataGrid_N = 4

    win = MyWindow(filename, strataGrid_N, alpha, MOE, e_moe, useBothAllocations=True)

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

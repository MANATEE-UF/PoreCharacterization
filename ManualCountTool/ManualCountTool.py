from tkinter import image_types
import numpy as np
from skimage import io
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import time
import os
import copy

class Pixel:
    def __init__(self, r, g, b, row, col):
        self.r = r
        self.g = g
        self.b = b
        self.row = row
        self.col = col
        self.color = True if self.r != self.g or self.r != self.b or self.b != self.g else False
    
    def ChangeColor(self, rgb):
        self.r = rgb[0]
        self.g = rgb[1]
        self.b = rgb[2]
    
    def ComparePixel(self, pixel):
        if self.r == pixel.r and self.g == pixel.g and self.b == pixel.b:
            return True
        else:
            return False
    
    def CompareColor(self, rgb):
        if self.r == rgb[0] and self.g == rgb[1] and self.b == rgb[2]:
            return True
        else:
            return False

class PixelMap:
    def __init__(self,image):
        self.rows = len(image)
        self.cols = len(image[0])
        self.pixels = np.empty((self.rows, self.cols), dtype=object)
        self.gridCenters = []
        self.gridCenters2D = None
        self.gridCenters3D = None
        self.actualNumPoints = 0

        if np.max(image) <= 1.0:
            image *= 255

        for i in range(self.rows):
            for j in range(self.cols):
                self.pixels[i][j] = Pixel(image[i][j], image[i][j], image[i][j], i, j)
    
    def SetGridPositions(self, numGridPoints, numGrids):
        numTranslations = numGrids - 1

        # Use heightToWidth ratio to ensure that grid is equally spaced and square according to ASTM standard
        heightToWidthRatio = self.rows / self.cols
        numGridRows = int(np.sqrt(heightToWidthRatio * numGridPoints))
        numGridCols = int(numGridPoints / numGridRows)

        print(f"Requested Number of Points: {numGridPoints}")
        print(f"Actual Number of Points: {numGridRows*numGridCols}")
        self.actualNumPoints = numGridRows*numGridCols

        colRegionBorders = np.linspace(0, self.cols, numGridCols+1, endpoint=True, dtype=int)
        rowRegionBorders = np.linspace(0, self.rows, numGridRows+1, endpoint=True, dtype=int)

        rowPositions = []
        colPositions = []
        for i in range(len(colRegionBorders)-1):
            colPositions.append(int((colRegionBorders[i]+colRegionBorders[i+1])/2))

        for i in range(len(rowRegionBorders)-1):
            rowPositions.append(int((rowRegionBorders[i]+rowRegionBorders[i+1])/2))

        self.gridCenters = []
        for row in rowPositions:
            for col in colPositions:
                self.gridCenters.append(self.pixels[row][col])

        gridCenters = np.array(self.gridCenters)
        self.gridCenters2D = gridCenters.reshape(numGridRows, numGridCols)

        maxXTranslation, maxYTranslation = self.__GetMaxTranslation__()

        randomNegative = lambda: 1 if np.random.random() < 0.5 else -1

        self.gridCenters3D = [copy.deepcopy(self.gridCenters2D)]
        
        for i in range(numTranslations):
            temp = copy.deepcopy(self.gridCenters2D)
            xTranslation = np.random.randint(2, maxXTranslation-3) * randomNegative()
            yTranslation = np.random.randint(2, maxYTranslation-3) * randomNegative()
            for row in range(len(self.gridCenters2D)):
                for col in range(len(self.gridCenters2D[row])):
                    temp[row][col].row += yTranslation
                    temp[row][col].col += xTranslation
            self.gridCenters3D.append(copy.deepcopy(temp))
        
        self.gridCenters3D = np.array(self.gridCenters3D)
  
    def __GetMaxTranslation__(self):
        topLeftGridPixel = self.gridCenters2D[0][0]
        topRightGridPixel = self.gridCenters2D[0][-1]
        bottomLeftGridPixel = self.gridCenters2D[-1][0]
        bottomRightGridPixel = self.gridCenters2D[-1][-1]

        # top(u) bottom(b) left(l) right(r) distance (d) from y/x border
        tldy = topLeftGridPixel.row
        tldx = topLeftGridPixel.col

        trdy = topRightGridPixel.row
        trdx = self.cols - topRightGridPixel.col

        bldy = self.rows - bottomLeftGridPixel.row
        bldx = bottomLeftGridPixel.col

        brdy = self.rows - bottomRightGridPixel.row
        brdx = self.cols - bottomRightGridPixel.col

        # distance between grid points
        griddx = int((self.gridCenters2D[0][1].col - self.gridCenters2D[0][0].col) / 2)
        griddy = int((self.gridCenters2D[1][0].row - self.gridCenters2D[0][0].row) / 2)

        maxYTranslation = np.min([tldy, trdy, bldy, brdy, griddy]) - 1
        maxXTranslation = np.min([tldx, trdx, bldx, brdx, griddx]) - 1

        return maxXTranslation, maxYTranslation
    
    # Overlay grid crosses onto pixel map
    def OverlayGrid(self):
        gridCrossColor = (0,255,255)

        for i in range(len(self.gridCenters3D)):
            for j in range(len(self.gridCenters3D[i])):
                for gridCenterPixel in self.gridCenters3D[i][j]:
                    pixelRow = gridCenterPixel.row
                    pixelCol = gridCenterPixel.col
                    # Center
                    self.pixels[pixelRow][pixelCol].ChangeColor(gridCrossColor)

                    # Three above
                    self.pixels[pixelRow-1][pixelCol].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow-2][pixelCol].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow-3][pixelCol].ChangeColor(gridCrossColor)

                    # Three below
                    self.pixels[pixelRow+1][pixelCol].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow+2][pixelCol].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow+3][pixelCol].ChangeColor(gridCrossColor)

                    # Three right
                    self.pixels[pixelRow][pixelCol+1].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow][pixelCol+2].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow][pixelCol+3].ChangeColor(gridCrossColor)

                    # Three left
                    self.pixels[pixelRow][pixelCol-1].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow][pixelCol-2].ChangeColor(gridCrossColor)
                    self.pixels[pixelRow][pixelCol-3].ChangeColor(gridCrossColor)

    # Shows full image
    def ShowImage(self):
        newImage = np.empty((self.rows,self.cols,3), dtype=int)
        for i in range(self.rows):
            for j in range(self.cols):
                newImage[i][j][0] = self.pixels[i][j].r
                newImage[i][j][1] = self.pixels[i][j].g
                newImage[i][j][2] = self.pixels[i][j].b
        plt.imshow(newImage)
        plt.show()

    # Save pixel map image
    def SaveImage(self):
        saveArray = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                saveArray[i][j][0] = self.pixels[i][j].r / 255
                saveArray[i][j][1] = self.pixels[i][j].g / 255
                saveArray[i][j][2] = self.pixels[i][j].b / 255
        
        plt.imsave("test.png", saveArray)

    # Initiate grid color change
    def ChangeGridColor(self, poreIndex, gridIndex, newColor):
        center = self.gridCenters3D[gridIndex].flatten()[poreIndex]
        pixelRow = center.row
        pixelCol = center.col

        # Center
        self.pixels[pixelRow][pixelCol].ChangeColor(newColor)

        # Three above
        self.pixels[pixelRow-1][pixelCol].ChangeColor(newColor)
        self.pixels[pixelRow-2][pixelCol].ChangeColor(newColor)
        self.pixels[pixelRow-3][pixelCol].ChangeColor(newColor)

        # Three below
        self.pixels[pixelRow+1][pixelCol].ChangeColor(newColor)
        self.pixels[pixelRow+2][pixelCol].ChangeColor(newColor)
        self.pixels[pixelRow+3][pixelCol].ChangeColor(newColor)

        # Three right
        self.pixels[pixelRow][pixelCol+1].ChangeColor(newColor)
        self.pixels[pixelRow][pixelCol+2].ChangeColor(newColor)
        self.pixels[pixelRow][pixelCol+3].ChangeColor(newColor)

        # Three left
        self.pixels[pixelRow][pixelCol-1].ChangeColor(newColor)
        self.pixels[pixelRow][pixelCol-2].ChangeColor(newColor)
        self.pixels[pixelRow][pixelCol-3].ChangeColor(newColor)

    # Returns np array containing image around grid center based on index
    # Always center around original grid index (0) to prevent view from jumping around too much
    def GetGridCenterImage(self, poreIndex, gridIndex, numSurroundingPixels=50):
        self.ChangeGridColor(poreIndex, gridIndex, (0,0,255))

        center = self.gridCenters3D[0].flatten()[poreIndex]

        newImage = np.zeros((numSurroundingPixels * 2 + 1, numSurroundingPixels * 2 + 1, 3), dtype=int)
        for i in range(numSurroundingPixels * 2 + 1):
            for j in range(numSurroundingPixels * 2 + 1):
                if 0 <= center.row - numSurroundingPixels + i < self.rows and 0 <= center.col - numSurroundingPixels + j < self.cols:
                    newImage[i][j][0] = self.pixels[center.row - numSurroundingPixels + i][center.col - numSurroundingPixels + j].r
                    newImage[i][j][1] = self.pixels[center.row - numSurroundingPixels + i][center.col - numSurroundingPixels + j].g
                    newImage[i][j][2] = self.pixels[center.row - numSurroundingPixels + i][center.col - numSurroundingPixels + j].b
        
        return newImage

    # Displays a single grid center image based on index
    def ShowGridCenter(self, index):
        center = self.gridCenters3D[0].flatten()[index]

        # Set center of interest to red
        center.r = 255
        center.g = 0
        center.b = 0
        newImage = self.GetGridCenterImage(index)
        
        plt.imshow(newImage)
        plt.show()

    # Scrolls through all grid centers found in image
    def ShowAllGridCenters(self):
        for gridIndex in range(len(self.gridCenters)):
            self.ChangeGridColor(gridIndex, (255, 0, 0))

            newImage = self.GetGridCenterImage(gridIndex)
            
            plt.imshow(newImage) 
            plt.show(block=False)
            plt.pause(0.5)
            plt.close("all")

            self.ChangeGridColor(gridIndex, (0, 255, 0))


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MyWindow(QMainWindow):

    def __init__(self, imagePath, numGridPoints, numGrids):
        super(MyWindow,self).__init__()

        self.setWindowTitle("Pore Counter Tool")

        self.sc = MplCanvas(self, width=7, height=7, dpi=100)

        vbox = QtWidgets.QVBoxLayout()

        hbox2 = QtWidgets.QHBoxLayout()

        self.lastEntryText = QtWidgets.QLabel("Last Entry: --")
        self.lastEntryText.setAlignment(QtCore.Qt.AlignCenter)
        self.lastEntryText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        self.indexProgressText = QtWidgets.QLabel(f"Pore: -/-, Grid: -/-")
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

        self.widget = QtWidgets.QWidget()
        self.widget.setLayout(vbox)
        self.setCentralWidget(self.widget)
        
        self.show()
        
        self.imageName = imagePath
        
        self.numGrids = numGrids
        self.InitializeImage(imagePath, numGridPoints, numGrids)
        self.numSurroundingPixels = np.max(self.myMap.__GetMaxTranslation__())

        self.poreIndex = 0
        self.gridIndex = 0
        self.numGridRows = len(self.myMap.gridCenters2D)

        newImage = self.myMap.GetGridCenterImage(self.poreIndex, self.gridIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

        self.widget.setFocus(QtCore.Qt.NoFocusReason)
    
    def ZoomOut(self):
        if self.numSurroundingPixels < 300:
            self.numSurroundingPixels += 25
        
        newImage = self.myMap.GetGridCenterImage(self.poreIndex, self.gridIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

        if self.numSurroundingPixels >= 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)

        if self.numSurroundingPixels <= 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)
        
        self.widget.setFocus(QtCore.Qt.NoFocusReason)
    
    def ZoomIn(self):
        if self.numSurroundingPixels > 25:
            self.numSurroundingPixels -= 25
        
        newImage = self.myMap.GetGridCenterImage(self.poreIndex, self.gridIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

        if self.numSurroundingPixels == 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)

        if self.numSurroundingPixels == 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)
        
        self.widget.setFocus(QtCore.Qt.NoFocusReason)

    def InitializeImage(self,imagePath, numGridPoints, numGrids):
        image = io.imread(imagePath, as_gray=True) # Enforces gray scale to ensure pixel map read consistently
        self.myMap = PixelMap(image)
        self.myMap.SetGridPositions(numGridPoints, numGrids)
        self.myMap.OverlayGrid()
        # self.myMap.SaveImage()
        # self.myMap.FindGridCenters()
        self.poreData = np.zeros((numGrids,self.myMap.actualNumPoints))
        self.indexProgressText.setText(f"Pore: 1/{self.myMap.actualNumPoints}, Grid: 1/{self.numGrids}")

    def keyPressEvent(self, event):
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
        if value == -1 and self.gridIndex > 0:
            self.myMap.ChangeGridColor(self.poreIndex, self.gridIndex, (0,255,255))
            self.gridIndex -= 1
        elif value == -1 and self.gridIndex == 0 and self.poreIndex > 0:
            self.myMap.ChangeGridColor(self.poreIndex, self.gridIndex, (0,255,255))
            self.poreIndex -= 1
            self.gridIndex = self.numGrids-1
        elif value == -1 and self.gridIndex == 0 and self.poreIndex == 0:
            pass
        else:
            self.myMap.ChangeGridColor(self.poreIndex, self.gridIndex, (0,255,0))
            self.poreData[self.gridIndex][self.poreIndex] = value

            # At last grid index, move to next pore index
            if self.gridIndex == self.numGrids-1:
                self.gridIndex = 0
                self.poreIndex += 1            
            else:
                self.gridIndex += 1

        if self.poreIndex >= self.myMap.actualNumPoints:
            porosity = []
            with open(f"{os.path.splitext(os.path.basename(self.imageName))[0]}_{len(self.myMap.gridCenters)}GridPoints_countData.csv", "a") as f:
                for i, grid in enumerate(self.poreData):
                    porosity.append((np.sum(self.poreData[i]) / self.myMap.actualNumPoints))
                    np.savetxt(f, 
                        grid.reshape((len(self.myMap.gridCenters2D), len(self.myMap.gridCenters2D[0]))), 
                        fmt="%.2f", 
                        delimiter=",", 
                        header=f"{os.path.splitext(os.path.basename(self.imageName))[0]} with {len(self.myMap.gridCenters)} grid points \n Grid {i} \n",
                        footer=f"\n Porosity: {porosity[i] * 100}%")
                
                    print(f"Grid {i} Porosity: {porosity[i]*100:.2f}%")
            print(f"Average Porosity: {np.mean(porosity) * 100:.2f}")
            print(f"Standard Deviation: {np.std(porosity) * 100:.2f}")
            quit()

        newImage = self.myMap.GetGridCenterImage(self.poreIndex, self.gridIndex, self.numSurroundingPixels)
        self.indexProgressText.setText(f"Pore: {self.poreIndex+1}/{self.myMap.actualNumPoints}, Grid: {self.gridIndex+1}/{self.numGrids}")

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Adjust the file name and number of grid points as needed
    filename = r"Radius1_Area1_Image6.tif"
    numberOfGridPoints = 100
    numGrids = 3
    win = MyWindow(filename, numberOfGridPoints, numGrids)

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

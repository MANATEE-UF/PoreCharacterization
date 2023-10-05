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

        for i in range(self.rows):
            for j in range(self.cols):
                self.pixels[i][j] = Pixel(image[i][j], image[i][j], image[i][j], i, j)
    
    # Overlay grid crosses onto pixel map
    def OverlayGrid(self, numGridPoints):
        heightToWidthRatio = self.rows / self.cols
        numGridRows = int(np.sqrt(heightToWidthRatio * numGridPoints))
        numGridCols = int(numGridPoints / numGridRows)

        print(f"Requested Number of Points: {numGridPoints}")
        print(f"Actual Number of Points: {numGridRows*numGridCols}")

        colPositions = np.linspace(0, self.cols, numGridCols+1, endpoint=False, dtype=int)[1:]
        rowPositions = np.linspace(0, self.rows, numGridRows+1, endpoint=False, dtype=int)[1:]

        gridCrossColor = (0,255,255)
        gridCenters = []
        for rowPos in rowPositions:
            for colPos in colPositions:
                self.gridCenters.append(self.pixels[rowPos][colPos])

                # Center
                self.pixels[rowPos][colPos].ChangeColor(gridCrossColor)

                # Three above
                self.pixels[rowPos-1][colPos].ChangeColor(gridCrossColor)
                self.pixels[rowPos-2][colPos].ChangeColor(gridCrossColor)
                self.pixels[rowPos-3][colPos].ChangeColor(gridCrossColor)

                # Three below
                self.pixels[rowPos+1][colPos].ChangeColor(gridCrossColor)
                self.pixels[rowPos+2][colPos].ChangeColor(gridCrossColor)
                self.pixels[rowPos+3][colPos].ChangeColor(gridCrossColor)

                # Three right
                self.pixels[rowPos][colPos+1].ChangeColor(gridCrossColor)
                self.pixels[rowPos][colPos+2].ChangeColor(gridCrossColor)
                self.pixels[rowPos][colPos+3].ChangeColor(gridCrossColor)

                # Three left
                self.pixels[rowPos][colPos-1].ChangeColor(gridCrossColor)
                self.pixels[rowPos][colPos-2].ChangeColor(gridCrossColor)
                self.pixels[rowPos][colPos-3].ChangeColor(gridCrossColor)
        
        gridCenters = np.array(self.gridCenters)
        self.gridCenters2D = gridCenters.reshape(numGridRows, numGridCols)
                
    # Find center of grid crosses
    # only works if grid crosses are 1 pixel wide
    def FindGridCenters(self):
        # Locate all grid centers by finding non-gray pixels
        for pixel in self.pixels.flatten():
            if pixel.color:
                if self.pixels[pixel.row - 1][pixel.col].color and self.pixels[pixel.row][pixel.col + 1].color:
                    self.gridCenters.append(pixel)
        
        # Reshape into 2D array
        gridCenters = np.array(self.gridCenters)
        cnt = 0
        lastRow = self.gridCenters[0].row
        for center in self.gridCenters:
            if center.row != lastRow:
                numCols = cnt
                break
            cnt += 1
        numRows = int(len(self.gridCenters) / numCols)
        self.gridCenters2D = gridCenters.reshape(numRows, numCols)

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

    # Recursive function to change color
    # Changes all connecting pixel matching color of base pixel
    def ChangeConnectingColor(self, row, col, newColor):
        currentPixel = self.pixels[row][col]

        baseColor = (currentPixel.r, currentPixel.g, currentPixel.b)

        # Check if pixel is already the requested newColor
        if currentPixel.CompareColor(newColor):
            return

        currentPixel.ChangeColor(newColor)

        neighbors = [self.pixels[row-1][col], self.pixels[row+1][col], self.pixels[row][col-1], self.pixels[row][col+1]]
        
        for neighbor in neighbors: 
            if neighbor.CompareColor(baseColor):
                self.ChangeConnectingColor(neighbor.row, neighbor.col, newColor)

    def ChangeGridColor(self, index, newColor):
        center = self.gridCenters[index]
        self.ChangeConnectingColor(center.row, center.col, newColor)

    # Returns np array containing image around grid center based on index
    def GetGridCenterImage(self, index, numSurroundingPixels=50):
        center = self.gridCenters[index]

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
        center = self.gridCenters[index]

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

    def __init__(self, imagePath, numGridPoints):
        super(MyWindow,self).__init__()

        self.setWindowTitle("Pore Counter Tool")

        self.sc = MplCanvas(self, width=7, height=7, dpi=100)

        vbox = QtWidgets.QVBoxLayout()

        hbox2 = QtWidgets.QHBoxLayout()

        self.lastEntryText = QtWidgets.QLabel("Last Data Entry: --")
        self.lastEntryText.setAlignment(QtCore.Qt.AlignCenter)
        self.lastEntryText.setStyleSheet("background-color: light gray; border: 1px solid black;")

        hbox2.addWidget(self.lastEntryText)
        
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

        self.InitializeImage(imagePath, numGridPoints)
        self.numSurroundingPixels = 50

        self.nextIndex = 0
        self.numGridRows = len(self.myMap.gridCenters2D)

        newImage = self.myMap.GetGridCenterImage(self.nextIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

        self.widget.setFocus(QtCore.Qt.NoFocusReason)
    
    def ZoomOut(self):
        if self.numSurroundingPixels < 300:
            self.numSurroundingPixels += 25
        
        newImage = self.myMap.GetGridCenterImage(self.nextIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

        if self.numSurroundingPixels == 300:
            self.zoomOutButton.setEnabled(False)
        else:
            self.zoomOutButton.setEnabled(True)

        if self.numSurroundingPixels == 25:
            self.zoomInButton.setEnabled(False)
        else:
            self.zoomInButton.setEnabled(True)
        
        self.widget.setFocus(QtCore.Qt.NoFocusReason)
    
    def ZoomIn(self):
        if self.numSurroundingPixels > 25:
            self.numSurroundingPixels -= 25
        
        newImage = self.myMap.GetGridCenterImage(self.nextIndex, self.numSurroundingPixels)

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

    def InitializeImage(self,imagePath, numGridPoints):
        image = io.imread(imagePath, as_gray=True) # Enforces gray scale to ensure pixel map read consistently
        image *= 255
        self.myMap = PixelMap(image)
        self.myMap.OverlayGrid(numGridPoints)
        # self.myMap.FindGridCenters()
        self.poreData = np.zeros(len(self.myMap.gridCenters))

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
        if value == -1 and self.nextIndex > 0:
            self.myMap.ChangeGridColor(self.nextIndex, (255,255,0))
            self.nextIndex -= 1
        elif value == -1 and self.nextIndex == 0:
            self.nextIndex = 0
        else:
            self.poreData[self.nextIndex] = value
            self.myMap.ChangeGridColor(self.nextIndex, (0,255,0))
            self.nextIndex += 1
        
        if self.nextIndex >= len(self.myMap.gridCenters):
            self.poreData = self.poreData.reshape(len(self.myMap.gridCenters2D), len(self.myMap.gridCenters2D[0]))
            np.savetxt(f"{os.path.splitext(os.path.basename(self.imageName))[0]}_{len(self.myMap.gridCenters)}GridPoints_countData.csv", 
                       self.poreData, 
                       fmt="%.2f", 
                       delimiter=",", 
                       header=f"{os.path.splitext(os.path.basename(self.imageName))[0]} with {len(self.myMap.gridCenters)} grid points \n",
                       footer=f"\n Porosity: {np.sum(self.poreData) / (np.shape(self.poreData)[0] * np.shape(self.poreData)[1]) * 100}")
            print(f"Porosity: {np.sum(self.poreData) / (np.shape(self.poreData)[0] * np.shape(self.poreData)[1]) * 100}")
            quit()

        newImage = self.myMap.GetGridCenterImage(self.nextIndex, self.numSurroundingPixels)

        self.sc.axes.cla()
        self.sc.axes.imshow(newImage)
        self.sc.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Adjust the file name and number of grid points as needed
    filename = "ManualCountTool/TestData/SE-28C-OuterBand-TestImage.tif"
    numberOfGridPoints = 100
    win = MyWindow(filename, numberOfGridPoints)

    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

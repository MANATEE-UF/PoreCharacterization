import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json
from tqdm import tqdm

def main():

    scale = 20/161 # For 760kx, nm/pixel
    areaScale = scale**2 # nm^2 / pixel^2

    jsonFile = open("UMo_GBS/gbs.json")

    data = json.load(jsonFile)
    imagesData = data["images"]
    annotationData = data["annotations"]

    areas = {"A":[], "C":[], "D":[], "E":[]}
    totalAreas = []
    for i, bubble in enumerate(tqdm(annotationData, desc="Calculating Bubble Areas")):
        bubbleVertices = np.array(bubble["segmentation"][0])
        bubbleVerticesX = bubbleVertices[0::2]
        bubbleVerticesY = bubbleVertices[1::2]
        sum = 0
        for j in range(len(bubbleVerticesX)-1):
            sum += ((bubbleVerticesX[j] * bubbleVerticesY[j+1]) - (bubbleVerticesX[j+1] * bubbleVerticesY[j]))
        sum += ((bubbleVerticesX[-1] * bubbleVerticesY[0]) - (bubbleVerticesX[0] * bubbleVerticesY[-1]))
        sum = np.abs(sum) 
        sum /= 2

        gridID = imagesData[bubble["image_id"]]["file_name"].split("_")[1]

        areas[gridID].append(sum * areaScale)
        totalAreas.append(sum * areaScale)

    # Area Box Plot
    stacked = False
    if stacked:
        totalAreas = []
        totalAreas.append(areas["A"])
        totalAreas.append(areas["C"])
        totalAreas.append(areas["D"])
        totalAreas.append(areas["E"])
        plt.hist(totalAreas, bins=100, histtype="bar", stacked=True, label=["Post A", "Post C", "Post D", "Post E"])
    else:
        plt.hist(areas["A"], label="Post A", bins=100, alpha=0.5)
        plt.hist(areas["C"], label="Post C", bins=100, alpha=0.5)
        plt.hist(areas["D"], label="Post D", bins=100, alpha=0.5)
        plt.hist(areas["E"], label="Post E", bins=100, alpha=0.5)
    
    plt.legend()
    plt.xlabel("Area (nm^2)")
    plt.ylabel("Frequency")
    plt.title("GBS Area Distribution")

    plt.show()
    

if __name__ == "__main__":
    main()
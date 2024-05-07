import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json
from tqdm import tqdm
import scipy

def main():

    scale = 20/161 # For 760kx, nm/pixel
    areaScale = scale**2 # nm^2 / pixel^2

    # Read 4039 data (low burnup)
    jsonFile = open("UMo_GBS/KGT4039.json")

    data = json.load(jsonFile)
    imagesData = data["images"]
    annotationData = data["annotations"]

    areas = {"4039" : [], "4039 A":[], "4039 C":[], "4039 D":[], "4039 E":[], 
             "4061" : [], "4061 A":[], "4061 B":[], "4061 C":[], "4061 D":[]}
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

        area = sum * areaScale

        # Trim outliers, bad segmentation
        if area > 35:
            continue

        areas["4039"].append(area)
        areas[f"4039 {gridID}"].append(area)
        totalAreas.append(area)

    # Read 4061 data (high burnup)
    jsonFile = open("UMo_GBS/KGT4061.json")

    data = json.load(jsonFile)
    imagesData = data["images"]
    annotationData = data["annotations"]

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

        gridID = imagesData[bubble["image_id"]]["file_name"].split("_")[0]

        area = sum * areaScale

        # Trim outliers, bad segmentation
        if area > 35:
            continue

        areas["4061"].append(area)
        areas[f"4061 {gridID}"].append(area)
        totalAreas.append(area)

    # indexing along 
    areaByPost = []
    areaByPostLabel = ["4039 A", "4039 C", "4039 D", "4039 E", "4061 A", "4061 B", "4061 C", "4061 D"]
    for label in areaByPostLabel:
        areaByPost.append(areas[label])

    # indexing along first axis gives the area distribution for a given burnup
    areaByBurnup = []
    areaByBurnupLabel = ["4039", "4061"]
    areaByBurnup.append(areas["4039"])
    areaByBurnup.append(areas["4061"])

    fig, axs = plt.subplots(2,2)

    axs[0][0].hist(areaByPost, bins=100, histtype="bar", stacked=True, label=areaByPostLabel)
    axs[1][0].hist(areaByBurnup, bins=100, histtype="bar", stacked=True, label=areaByBurnupLabel)
    axs[0][1].hist(areas["4039 A"], label="4039 A", bins=100, alpha=0.5)
    axs[0][1].hist(areas["4039 C"], label="4039 C", bins=100, alpha=0.5)
    axs[0][1].hist(areas["4039 D"], label="4039 D", bins=100, alpha=0.5)
    axs[0][1].hist(areas["4039 E"], label="4039 E", bins=100, alpha=0.5)
    axs[0][1].hist(areas["4061 A"], label="4061 A", bins=100, alpha=0.5)
    axs[0][1].hist(areas["4061 B"], label="4061 B", bins=100, alpha=0.5)
    axs[0][1].hist(areas["4061 C"], label="4061 C", bins=100, alpha=0.5)
    axs[0][1].hist(areas["4061 D"], label="4061 D", bins=100, alpha=0.5)

    axs[1][1].hist(areaByBurnup[0], label="4039", bins=100, alpha=0.5)
    axs[1][1].hist(areaByBurnup[1], label="4061", bins=100, alpha=0.5)

    axs[0][0].legend()
    axs[0][1].legend()
    axs[1][0].legend()
    axs[1][1].legend()
    axs[1][0].set_xlabel("Area (nm^2)")
    axs[1][1].set_xlabel("Area (nm^2)")
    axs[0][0].set_ylabel("Frequency")
    axs[1][0].set_ylabel("Frequency")
    plt.title("GBS Area Distribution")

    plt.show()

    distribution1 = "4039 A"
    distribution2 = "4039 D"
    altHyp = "lesser"
    print()
    print(f"K-S test between {distribution1} and {distribution2}")
    ksTest = scipy.stats.ks_2samp(areas[distribution1], areas[distribution2], alternative=altHyp)
    if altHyp == "two-sided":
        if ksTest.pvalue < 0.05:
            print(f"{distribution1} and {distribution2} do not have identical distribution")
        else:
            print(f"{distribution1} and {distribution2} have identical distribution")
    else:
        if ksTest.pvalue < 0.05:
            print(f"{distribution2} is {altHyp} than {distribution1}")
        else:
            print(f"{distribution2} is not {altHyp} than {distribution1}")
    print(f"Test statistic: {ksTest.statistic}")
    print(f"p-value: {ksTest.pvalue}")
    print()
    

if __name__ == "__main__":
    main()
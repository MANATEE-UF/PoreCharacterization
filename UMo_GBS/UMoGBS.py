import numpy as np
import skimage
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def DistanceEq2D(p1, p2):
    return math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

def SlopeEq2D(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

# Return angle between vectors centered at v0 in degrees
def AngleBetweenVectors(v0, v1, v2):
    d1 = DistanceEq2D(v0, v1)
    d2 = DistanceEq2D(v0, v2)

    dotProduct = (v1[0]-v0[0]) * (v2[0]-v0[0]) + (v1[1]-v0[1]) * (v2[1]-v0[1])
    try:
        theta = math.acos(dotProduct / (d1 * d2))
    except ValueError:
        return 180
    theta = math.degrees(theta)

    return theta

def GetLatticeParameters(spotPositions, scaleFactor=None):
    params = []
    spotPositionsList = spotPositions.tolist()
    for centralPoint in spotPositionsList:
        tempPoints = spotPositionsList
        tempPoints.remove(centralPoint)

        # Find point closest to central point
        firstPoint = None
        minDist = 1E9
        for point in tempPoints:
            dist = DistanceEq2D(point, centralPoint)
            if dist < minDist:
                minDist = dist
                firstPoint = point
        tempPoints.remove(firstPoint)

        # Pick the second closest point that passes lattice checks
        secondPoint = None
        minDist = 1E9
        for point in tempPoints:
            if AngleBetweenVectors(centralPoint, firstPoint, point) > 150 or AngleBetweenVectors(centralPoint, firstPoint, point) < 20:
                continue
            dist = DistanceEq2D(point, centralPoint)

            if dist < minDist:
                minDist = dist
                secondPoint = point

        # Find a and b distances
        dMin = min(DistanceEq2D(centralPoint, firstPoint), DistanceEq2D(centralPoint, secondPoint))
        dMax = max(DistanceEq2D(centralPoint, firstPoint), DistanceEq2D(centralPoint, secondPoint))

        # Find angle between vectors, normalize to [0,2pi]
        theta = AngleBetweenVectors(centralPoint, firstPoint, secondPoint)
        theta = theta/2 if theta > 100 else theta # FIXME: This is really hacky for handling hexagons

        params.append([dMin, dMax, theta])
    
    try:
        medianVals = list(np.median(params, 0))
    except:
        print("Error with values:")
        print(params)
        return None
    
    # Convert from pixel to A^-1 distance
    if scaleFactor is not None:
        medianVals[0] *= scaleFactor
        medianVals[1] *= scaleFactor

    return medianVals

def main():
    # Use BF images to get GBS bubble positions
    scaleKey = {"2.15Mx":10/113, "1.5Mx":10/161, "1.05Mx":20/224, "760kx":20/161, "540kx":50/281, "380kx": 50/200, "190kx": 100/200}
    imagePath = "Loc41_540kx.png"
    zoomLevel = "540kx"

    # Read image and plot unedited
    img = skimage.util.img_as_float(skimage.io.imread(imagePath, as_gray=True))
    img = skimage.util.crop(img, [0,80]) # Crop scale bar on bottom

    # Apply otsu threshold, remove small objects, remove small holes, and remove labels that touch the image border
    # threshold = skimage.filters.threshold_sauvola(img, window_size=75)
    threshold = skimage.filters.threshold_mean(img)
    mask = img > threshold
    mask = skimage.morphology.remove_small_objects(mask, 50)
    mask = skimage.morphology.remove_small_holes(mask, 50)
    mask = skimage.segmentation.clear_border(mask)

    # Apply mask to generate labels
    labels = skimage.measure.label(mask)
    label_image = skimage.color.label2rgb(labels, image=img, bg_label=0)
    props = skimage.measure.regionprops(labels, img)
    
    plt.subplot(1,2,1)
    plt.imshow(label_image)

    areas = []
    eccentricities = []
    centroids = []
    diameters = []
    for i, prop in enumerate(props):
        y0, x0 = prop.centroid
        plt.text(x0, y0, str(i+1), va='center', ha="center", c="w", fontsize=12,path_effects=[pe.withStroke(linewidth=4, foreground="black")])

        areas.append(prop.area)
        eccentricities.append(prop.eccentricity)
        centroids.append((x0,y0))
        diameters.append(prop.equivalent_diameter_area)
    
    radii = np.array(diameters) * scaleKey[zoomLevel] / 2
    eccentricities = np.array(eccentricities)
    areas = np.array(areas) * scaleKey[zoomLevel]**2

    # Radii Box Subplot
    plt.subplot(3,2,2)
    bp = plt.boxplot(radii, vert=False)
    plt.xlabel("Radius (nm)")
    plt.title("Equivalent Area Radius")
    plt.yticks([])
    plt.minorticks_on()

    fliers = bp["fliers"][0]
    iqr = bp['boxes'][0]
    caps = bp["caps"]
    med = bp["medians"][0]

    yOffset = 0.3
    median  = med.get_xdata()[1]
    plt.text(median, 1-yOffset, f"Median: {median:.2f}", ha="center", fontsize=10, fontweight="bold")

    pc25 = iqr.get_xdata().min()
    pc75 = iqr.get_xdata().max()
    plt.text(pc25, 1+yOffset, f"Q1: {pc25:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.text(pc75, 1+yOffset, f"Q3: {pc75:.2f}", ha="center", fontsize=10, fontweight="bold")

    leftCap = caps[0].get_xdata()[0]
    rightCap = caps[1].get_xdata()[0]
    plt.text(leftCap, 1-yOffset, f"Bottom: {leftCap:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.text(rightCap, 1-yOffset, f"Top: {rightCap:.2f}", ha="center", fontsize=10, fontweight="bold")

    yOffset = 0.1
    for flier in fliers.get_xdata():
        idx = np.where(radii == flier)[0][0] + 1
        plt.text(flier, 1+yOffset, str(idx), ha="center", fontsize=8)
        yOffset *= -1

    # Eccentricity Box Subplot
    plt.subplot(3,2,4)
    bp = plt.boxplot(eccentricities, vert=False)
    plt.xlabel("Eccentricity")
    plt.title("Eccentricity")
    plt.yticks([])

    fliers = bp["fliers"][0]
    iqr = bp['boxes'][0]
    caps = bp["caps"]
    med = bp["medians"][0]

    yOffset = 0.3
    median  = med.get_xdata()[1]
    plt.text(median, 1-yOffset, f"Median: {median:.2f}", ha="center", fontsize=10, fontweight="bold")

    pc25 = iqr.get_xdata().min()
    pc75 = iqr.get_xdata().max()
    plt.text(pc25, 1+yOffset, f"Q1: {pc25:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.text(pc75, 1+yOffset, f"Q3: {pc75:.2f}", ha="center", fontsize=10, fontweight="bold")

    leftCap = caps[0].get_xdata()[0]
    rightCap = caps[1].get_xdata()[0]
    plt.text(leftCap, 1-yOffset, f"Bottom: {leftCap:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.text(rightCap, 1-yOffset, f"Top: {rightCap:.2f}", ha="center", fontsize=10, fontweight="bold")

    yOffset = 0.1
    for flier in fliers.get_xdata():
        idx = np.where(eccentricities == flier)[0][0] + 1
        plt.text(flier, 1+yOffset, str(idx), ha="center", fontsize=8)
        yOffset *= -1

    # Area Box Subplot
    plt.subplot(3,2,6)
    bp = plt.boxplot(areas, vert=False)
    plt.xlabel("Area (nm^2)")
    plt.title("Area")
    plt.yticks([])

    fliers = bp["fliers"][0]
    iqr = bp['boxes'][0]
    caps = bp["caps"]
    med = bp["medians"][0]

    yOffset = 0.3
    median  = med.get_xdata()[1]
    plt.text(median, 1-yOffset, f"Median: {median:.2f}", ha="center", fontsize=10, fontweight="bold")

    pc25 = iqr.get_xdata().min()
    pc75 = iqr.get_xdata().max()
    plt.text(pc25, 1+yOffset, f"Q1: {pc25:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.text(pc75, 1+yOffset, f"Q3: {pc75:.2f}", ha="center", fontsize=10, fontweight="bold")

    leftCap = caps[0].get_xdata()[0]
    rightCap = caps[1].get_xdata()[0]
    plt.text(leftCap, 1-yOffset, f"Bottom: {leftCap:.2f}", ha="center", fontsize=10, fontweight="bold")
    plt.text(rightCap, 1-yOffset, f"Top: {rightCap:.2f}", ha="center", fontsize=10, fontweight="bold")

    yOffset = 0.1
    for flier in fliers.get_xdata():
        idx = np.where(areas == flier)[0][0] + 1
        plt.text(flier, 1+yOffset, str(idx), ha="center", fontsize=8)
        yOffset *= -1
    # GetLatticeParameters(centroids, scaleFactor=scaleKey["380kx"])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
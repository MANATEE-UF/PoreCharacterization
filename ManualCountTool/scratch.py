import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import csv
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import pandas
import time
import seaborn as sns
import os
import tqdm
from skimage import io

def CompareTimes():
    start = time.time()
    n = 200
    p = 0.5
    for i in range(100000):
        binom = scipy.stats.binomtest(100, n, p)
        ci = binom.proportion_ci(0.95, method="wilson")
        lowerCL = ci.low
        upperCL = ci.high
        moe = upperCL - lowerCL
    print(time.time()-start)

    start = time.time()
    z = scipy.stats.norm.ppf(0.95)
    for i in range(100000):
        moe = (z * np.sqrt(n) / ((n+z**2)) * (p * (1-p) + (z**2 / (4*n)))**0.5)
    print(time.time()-start)

    for i in range(100000):
        lowerCL, upperCL = scipy.stats.binom.interval(0.95, n, p)
        moe = upperCL - lowerCL
    print(time.time()-start)


def PlotSamplesPerInitialGuess():
    porosityGuesses = np.linspace(0,1,21)
    porosityGuesses[0] = 0.01
    porosityGuesses[-1] = 0.99
    samplesNeeded = []
    maxIters = 50
    for i in range(21):
        porosityGuess = porosityGuesses[i]
        initialGuesses = np.ones(16) * porosityGuess
        highVarGuess = np.ones(16) * 0.5
        numStrata_N = 4
        N=256*256
        MOE = 0.05
        e_moe = 0.01
        alpha = 0.95
        print(f"{i+1}/21: {porosityGuess:.2f}")

        W_h = 1 / numStrata_N**2 # Value constant because image is split evenly. small differences neglected
        initialStrataProportion = porosityGuess
        variance = np.sum(W_h * np.sqrt(highVarGuess * (1 - highVarGuess)))**2 / numStrata_N**2 - ((1/N) * np.sum(W_h * highVarGuess * (1 - highVarGuess))) # highest variance given the initial guesses. ensures that first guess is very small
        if initialStrataProportion > 0.5 and MOE > 1-initialStrataProportion:
            MOE = ((1-initialStrataProportion)+MOE) / 2 # want to keep +- MOE as close as possible on the open side. so if p=0.01, with 5% MOE, the CI should be (0,0.06)
            # print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {MOE:.2f}")
        elif initialStrataProportion < 0.5 and MOE > initialStrataProportion:
            MOE = (initialStrataProportion + MOE) / 2
            # print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {initialStrataProportion:.2f}")
        upperCL = 1.0
        lowerCL = 0.0
        withinTolerance = False
        d = 0.5
        currentIter = 0
        while not withinTolerance and currentIter < maxIters:
            n = np.sum(W_h * np.sqrt(initialGuesses * (1-initialGuesses)) / variance) # smaller variance, higher n
            n = int(np.ceil(n))
            lowerCL, upperCL = TwoSidedCL_A(n, initialStrataProportion, alpha)
            lowerCL /= n
            upperCL /= n

            # FIXME: Assumes that variance will always be too high to start. sometimes it starts too low, then d gets too small over time and eventually maxiters is reached
            if ((upperCL - lowerCL) / 2) > MOE: # Eq.15 not satisfied
                variance *= d
            else: # Eq. 15 satisfied
                pctDiff = abs((((upperCL - lowerCL) / 2) - MOE) / MOE)
                if pctDiff > e_moe: # variance too low, overestimating how many sample points needed
                    variance /= d
                    d += (1-d)*0.2
                    variance *= d
                else:
                    withinTolerance = True
            currentIter += 1
        samplesNeeded.append(n)
        print(f"({lowerCL}, {upperCL})")
        print(f"MOE:{((upperCL - lowerCL) / 2):.3f}")
        print()
        
    plt.plot(porosityGuesses, samplesNeeded)
    plt.scatter(porosityGuesses, samplesNeeded)
    plt.show()

def PlotSampleFractionPerInitialConditions():
    MOEs = [0.05, 0.03]
    porosityGuesses = np.linspace(0,1,21)
    porosityGuesses[0] = 0.01
    porosityGuesses[-1] = 0.99
    cs = ["r","g","b"]
    for j in range(2):
        samplesNeeded = []
        maxIters = 10
        for i in range(11):
            porosityGuess = porosityGuesses[i]
            initialGuesses = np.ones(16) * porosityGuess
            highVarGuess = np.ones(16) * 0.5
            numStrata_N = 4
            N=256*256
            MOE = MOEs[j]
            e_moe = 0.1
            alpha = 0.95
            print(f"{i+1}/21: {100*porosityGuess:.2f}")

            W_h = 1 / numStrata_N**2 # Value constant because image is split evenly. small differences neglected
            initialStrataProportion = porosityGuess
            variance = np.sum(W_h * np.sqrt(highVarGuess * (1 - highVarGuess)))**2 / numStrata_N**2 - ((1/N) * np.sum(W_h * highVarGuess * (1 - highVarGuess))) # highest variance given the initial guesses. ensures that first guess is very small
            if initialStrataProportion > 0.5 and MOE > 1-initialStrataProportion:
                MOE = ((1-initialStrataProportion)+MOE) / 2 # want to keep +- MOE as close as possible on the open side. so if p=0.01, with 5% MOE, the CI should be (0,0.06)
                # print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {MOE:.2f}")
            elif initialStrataProportion < 0.5 and MOE > initialStrataProportion:
                MOE = (initialStrataProportion + MOE) / 2
                # print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {initialStrataProportion:.2f}")
            upperCL = 1.0
            lowerCL = 0.0
            withinTolerance = False
            d = 0.5
            currentIter = 0
            while not withinTolerance and currentIter < maxIters:
                n = np.sum(W_h * np.sqrt(initialGuesses * (1-initialGuesses)) / variance) # smaller variance, higher n
                n = int(np.ceil(n))
                lowerCL, upperCL = TwoSidedCL_A(n, initialStrataProportion, alpha)
                lowerCL /= n
                upperCL /= n

                if ((upperCL - lowerCL) / 2) > MOE: # Eq.15 not satisfied
                    variance *= d
                else: # Eq. 15 satisfied
                    pctDiff = abs((((upperCL - lowerCL) / 2) - MOE) / MOE)
                    if pctDiff > e_moe: # variance too low, overestimating how many sample points needed
                        variance /= d
                        d += (1-d)*0.2
                        variance *= d
                    else:
                        withinTolerance = True
                currentIter += 1
            samplesNeeded.append(n/N)
            print(f"({lowerCL*100:.2f}, {upperCL*100:.2f})")
            print(f"MOE:{100*((upperCL - lowerCL) / 2):.2f}")
            print()
        plt.plot(porosityGuesses[:11], samplesNeeded, c=cs[j], label=f"{MOE*100:.1f}% MOE")
        plt.scatter(porosityGuesses[:11], samplesNeeded, c=cs[j])
    plt.legend()
    plt.show()

def Plot_n_ratio():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times"
    })

    ratios = []
    W_h = 1/16

    initialGuesses = np.ones(16)
    initialGuesses[0:4] = 0.05
    initialGuesses[4:8] = 0.95
    initialGuesses[8:] = 0.5
    print(initialGuesses)
    ratio = W_h * (np.sum(np.sqrt(initialGuesses * (1-initialGuesses))))**2 / np.sum(initialGuesses * (1-initialGuesses))
    print(ratio)

    for i in range(100000):
        initialGuesses = np.random.choice([0.05, 0.125, 0.25, 0.375, 0.50, 0.675, 0.75, 0.875, 0.95], 16)

        ratio = W_h * (np.sum(np.sqrt(initialGuesses * (1-initialGuesses))))**2 / np.sum(initialGuesses * (1-initialGuesses))

        ratios.append(ratio)
    
    fig, axs = plt.subplots(1,2)
    

    bins = np.arange(0.85, 1.0, 0.01)
    axs[0].hist(ratios, bins=bins, edgecolor="black")
    axs[0].set_xlabel(r"$n_{opt} / n_{prop}$", fontsize=14)
    axs[0].set_ylabel("Frequency", fontsize=14)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0].set_xlim(0.85,1.0)
    axs[0].tick_params(axis="both", which="major", labelsize=12)

    axs[1].ecdf(ratios)
    axs[1].set_xlabel(r"$n_{opt} / n_{prop}$", fontsize=14)
    axs[1].set_ylabel("Probability of Occurence", fontsize=14)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1].set_xlim(0.85,1.0)
    axs[1].tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.savefig("OptimalAllocTheoreticalSavings.png", dpi=300)

def PlotSurface():
    x = [0.05, 0.125, 0.25, 0.375, 0.5]
    y = np.arange(100,1000, 1)

    numPoints = {0.05:100, 0.125:500, 0.25:1000, 0.375:1500, 0.5:2000}
    def func(x,y):
        return numPoints[x] / y**2
    
    XX,YY = np.meshgrid(x,y)

    zs = []
    for i in range(len(np.ravel(XX))):
        zs.append(func(np.ravel(XX)[i], np.ravel(YY)[i]))
    zs = np.array(zs)
    Z = zs.reshape(XX.shape)

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(Z)
    mappable.set_clim(0, max(zs+0.03)) # optional

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(XX*100, YY, Z, cmap=mappable.cmap, norm=mappable.norm, linewidth=0, antialiased=False)

    surf = ax.plot_surface(XX*100, YY, Z+0.03, cmap=mappable.cmap, norm=mappable.norm, linewidth=0, antialiased=False)

    ax.set_zlim(0,max(zs+0.05))
    ax.set_xlabel('Porosity (%)')
    ax.set_ylabel('Image Width (pixels)')
    ax.set_zlabel('Sampling Fraction')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def PlotContour():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    "font.size":14
    })

    x = [0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95]
    y = np.logspace(4,6.1, 20)

    numPoints = {0.05:100, 0.125:500, 0.25:1000, 0.375:1500, 0.5:1750, 0.625:1500, 0.75: 1000, 0.875:500, 0.95: 100}
    def func(x,y):
        return numPoints[x] / y
    
    XX,YY = np.meshgrid(x,y)

    zs = []
    for i in range(len(np.ravel(XX))):
        zs.append(func(np.ravel(XX)[i], np.ravel(YY)[i]))
    zs = np.array(zs)
    Z = zs.reshape(XX.shape)

    # plot contour
    plt.contourf(XX,YY,Z)
    plt.xlabel("Porosity")
    plt.ylabel("Number of Pixels")
    CS = plt.contour(XX, YY, Z, [0.01, 0.05, 0.10])
    plt.clabel(CS, colors="w")
    plt.yscale("log")

    # plot y-values where square micrograph sizes are
    x = [0.05,0.95]
    plt.plot(x, [256*256, 256*256], c="r", linestyle="--", linewidth=0.5)
    plt.plot(x, [512*512, 512*512], c="r", linestyle="--", linewidth=0.5)
    plt.plot(x, [1028*1028, 1028*1028], c="r", linestyle="--", linewidth=0.5)
    plt.text(0.1, 250*250, "256 x 256", bbox={"edgecolor":"r", "facecolor":"w"})
    plt.text(0.1, 500*500, "512 x 512", bbox={"edgecolor":"r", "facecolor":"w"})
    plt.text(0.1, 1000*1000, "1024 x 1024", bbox={"edgecolor":"r", "facecolor":"w"})

    plt.show()

def TwoSidedCL_A(n, p, alpha):
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

def TopOneSidedCL_A(n, p, alpha):
    A_U = n

    sum = 1.0
    A_L = 0

    while sum > alpha:
        A_L += 1
        sum = 0
        for i in range(A_L, n+1):
            sum += scipy.stats.binom.pmf(i, n, p)
    
    return A_L-1, A_U

def BottomOneSidedCL_A(n, p, alpha):
    sum = 0
    A_U = 0
    while sum <= alpha:
        A_U += 1
        sum = 0
        for i in range(A_U):
            sum += scipy.stats.binom.pmf(i, n, p)

    A_L = 0
    
    return A_L, A_U-1

def UpperCL_A(n, p, alpha):
    sum = 0
    A_U = 0
    while sum <= alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
        A_U += 1
        sum = 0
        for i in range(A_U):
            sum += scipy.stats.binom.pmf(i, n, p)
    
    return A_U-1

def LowerCL_A(n, p, alpha):
    sum = 1.0
    A_L = 0
    while sum > alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
        A_L += 1
        sum = 0
        for i in range(A_L, n):
            sum += scipy.stats.binom.pmf(i, n, p)
    
    return A_L-1

def PlotSimResiduals():
    plt.rc("axes", titlesize=20)
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    categoryDict = {"random":{"small":0, "medium":1, "large":2, "mixed":3}, "clustered":{"small":4, "medium":5, "large":6, "mixed":7}}
    colorDict = {0:"deeppink", 1:"orchid", 2:"lightsteelblue", 3:"cyan", 4:"springgreen", 5:"darkkhaki", 6:"darkorange", 7:"lightcoral"}
    titleDict = {0:"Random, Small", 1:"Random, Medium", 2:"Random, Large", 3:"Random, Mixed", 4:"Clustered, Small", 5:"Clustered, Medium", 6:"Clustered, Large", 7:"Clustered, Mixed"}

    fig,axs = plt.subplots(nrows=2, ncols=4)
    with open('key.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        cnt = 0
        for row in csv_reader:
            if cnt==0:
                cnt +=1
                continue

            idxToUse = categoryDict[row[4]][row[3]]
            unravledIdx = np.unravel_index(idxToUse, (2,4))

            axs[unravledIdx[0]][unravledIdx[1]].scatter(float(row[2]), float(row[6])-float(row[2]), c=colorDict[idxToUse], label=f"{row[4]}, {row[3]}", edgecolors="k",s=100)    
            axs[unravledIdx[0]][unravledIdx[1]].scatter(float(row[2]), float(row[10])-float(row[2]), c=colorDict[idxToUse], label=f"{row[4]}, {row[3]}", edgecolors="k",s=100, marker="^")    

    axs[1][0].set_xlabel("Porosity (%)")
    axs[1][1].set_xlabel("Porosity (%)")
    axs[1][2].set_xlabel("Porosity (%)")
    axs[1][3].set_xlabel("Porosity (%)")
    axs[0][0].set_ylabel("Measurement Residual (%)")
    axs[1][0].set_ylabel("Measurement Residual (%)")
    cnt = 0
    for ax in axs.flat:
        ax.hlines(5, 0, 55, color="r")
        ax.hlines(-5, 0, 55, color="r")
        ax.set_ylim(-11, 11)
        ax.set_xlim(0, 55)
        ax.set_title(titleDict[cnt])
        cnt += 1

    # plt.rc("axes", titlesize=30)
    # plt.rc("axes", labelsize=14)
    plt.show()

def PlotSimSuccesses2():
    plt.rcParams["font.family"] = "serif"
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)

    categoryDict = {"5%":{"Small":{"Random":0,"Clustered":1},"Medium":{"Random":2,"Clustered":3},"Large":{"Random":4,"Clustered":5},"Mixed":{"Random":6,"Clustered":7}}, 
                    "12.5%":{"Small":{"Random":8,"Clustered":9},"Medium":{"Random":10,"Clustered":11},"Large":{"Random":12,"Clustered":13},"Mixed":{"Random":14,"Clustered":15}}, 
                    "25%":{"Small":{"Random":16,"Clustered":17},"Medium":{"Random":18,"Clustered":19},"Large":{"Random":20,"Clustered":21},"Mixed":{"Random":22,"Clustered":23}}, 
                    "37.5%":{"Small":{"Random":24,"Clustered":25},"Medium":{"Random":26,"Clustered":27},"Large":{"Random":28,"Clustered":29},"Mixed":{"Random":30,"Clustered":31}}, 
                    "50%":{"Small":{"Random":32,"Clustered":33},"Medium":{"Random":34,"Clustered":35},"Large":{"Random":36,"Clustered":37},"Mixed":{"Random":38,"Clustered":39}}}

    counts = {"Optimal":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      "Proportional":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}

    with open('key.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        cnt = 0
        for row in csv_reader:
            if cnt==0:
                cnt +=1
                continue
            
            idx = categoryDict[row[1] + "%"][row[3]][row[4]]

            if float(row[7]) < float(row[2]) and float(row[2]) < float(row[8]):
                counts["Optimal"][idx] += 1

            if float(row[11]) < float(row[2]) and float(row[2]) < float(row[12]):
                counts["Proportional"][idx] += 1

    fig, ax = plt.subplots(layout='constrained')

    x = np.arange(40)  # the label locations
    offset = 0

    for attribute, measurement in counts.items():
        ax.scatter(x+offset, 100*np.array(measurement)/20, s=200, label=attribute, edgecolor="k")
        ax.plot(x+offset, 100*np.array(measurement)/20)
        offset += 0.1
    
    ax.set_xticks(x+0.05, np.arange(40))
    ax.set_yticks(np.arange(85,105,5))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of Measurements Within MOE (%)')
    ax.set_xlabel("Simulated Image Class ID")
    plt.legend(bbox_to_anchor=(1.01,0.5))
    
    ax.set_ylim(84, 101)

    plt.show()

def PlotSimResultsStrata():
    # First row: MOE for all 40 groups for two allocation methods for three MOEs (scatter)
    # Second row: Percent of successes for all 40 groups for two allocation methods for three MOEs (scatter)
    # Third row: Cost savings ratio for all 40 groups for three MOEs (violin)
    numGroups = 40

    plt.rcParams["font.family"] = "serif"
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)

    MOEs = {"Optimal 3x3" : np.zeros(numGroups),
            "Proportional 3x3": np.zeros(numGroups),
            "Optimal 4x4": np.zeros(numGroups),
            "Proportional 4x4": np.zeros(numGroups),
            "Optimal 5x5": np.zeros(numGroups),
            "Proportional 5x5": np.zeros(numGroups)}
    
    successes = {"Optimal 3x3" : np.zeros(numGroups),
            "Proportional 3x3": np.zeros(numGroups),
            "Optimal 4x4": np.zeros(numGroups),
            "Proportional 4x4": np.zeros(numGroups),
            "Optimal 5x5": np.zeros(numGroups),
            "Proportional 5x5": np.zeros(numGroups)}
    
    successesStd = {"Optimal 3x3" : np.zeros(numGroups),
            "Proportional 3x3": np.zeros(numGroups),
            "Optimal 4x4": np.zeros(numGroups),
            "Proportional 4x4": np.zeros(numGroups),
            "Optimal 5x5": np.zeros(numGroups),
            "Proportional 5x5": np.zeros(numGroups)}
    
    costSavingsRatio = {"3x3" : np.zeros(numGroups),
                        "4x4" : np.zeros(numGroups),
                        "5x5" : np.zeros(numGroups)}


    df_3x3 = pandas.read_csv("SimOutput_5MOE_3x3.csv", header=None)
    df_4x4 = pandas.read_csv("SimOutput_5MOE_4x4.csv", header=None)
    df_5x5 = pandas.read_csv("SimOutput_5MOE_5x5.csv", header=None)

    groupings = np.arange(0,410000,10000)
    numSamples = 10000
    for i in range(numGroups):
        MOEs["Optimal 3x3"][i] = np.average(df_3x3.iloc[groupings[i]:groupings[i+1], 10])
        MOEs["Optimal 4x4"][i] = np.average(df_4x4.iloc[groupings[i]:groupings[i+1], 10])
        MOEs["Optimal 5x5"][i] = np.average(df_5x5.iloc[groupings[i]:groupings[i+1], 10])
        MOEs["Proportional 3x3"][i] = np.average(df_3x3.iloc[groupings[i]:groupings[i+1], 15])
        MOEs["Proportional 4x4"][i] = np.average(df_4x4.iloc[groupings[i]:groupings[i+1], 15])
        MOEs["Proportional 5x5"][i] = np.average(df_5x5.iloc[groupings[i]:groupings[i+1], 15])

        successes["Optimal 3x3"][i] = 100 * np.average(np.logical_and(np.array(df_3x3.iloc[groupings[i]:groupings[i+1], 2] < df_3x3.iloc[groupings[i]:groupings[i+1], 9]), np.array(df_3x3.iloc[groupings[i]:groupings[i+1], 2] > df_3x3.iloc[groupings[i]:groupings[i+1], 8])))
        successes["Optimal 4x4"][i] = 100 * np.average(np.logical_and(np.array(df_4x4.iloc[groupings[i]:groupings[i+1], 2] < df_4x4.iloc[groupings[i]:groupings[i+1], 9]), np.array(df_4x4.iloc[groupings[i]:groupings[i+1], 2] > df_4x4.iloc[groupings[i]:groupings[i+1], 8])))
        successes["Optimal 5x5"][i] = 100 * np.average(np.logical_and(np.array(df_5x5.iloc[groupings[i]:groupings[i+1], 2] < df_5x5.iloc[groupings[i]:groupings[i+1], 9]), np.array(df_5x5.iloc[groupings[i]:groupings[i+1], 2] > df_5x5.iloc[groupings[i]:groupings[i+1], 8])))
        successes["Proportional 3x3"][i] = 100 * np.average(np.logical_and(np.array(df_3x3.iloc[groupings[i]:groupings[i+1], 2] < df_3x3.iloc[groupings[i]:groupings[i+1], 14]), np.array(df_3x3.iloc[groupings[i]:groupings[i+1], 2] > df_3x3.iloc[groupings[i]:groupings[i+1], 13])))
        successes["Proportional 4x4"][i] = 100 * np.average(np.logical_and(np.array(df_4x4.iloc[groupings[i]:groupings[i+1], 2] < df_4x4.iloc[groupings[i]:groupings[i+1], 14]), np.array(df_4x4.iloc[groupings[i]:groupings[i+1], 2] > df_4x4.iloc[groupings[i]:groupings[i+1], 13])))
        successes["Proportional 5x5"][i] = 100 * np.average(np.logical_and(np.array(df_5x5.iloc[groupings[i]:groupings[i+1], 2] < df_5x5.iloc[groupings[i]:groupings[i+1], 14]), np.array(df_5x5.iloc[groupings[i]:groupings[i+1], 2] > df_5x5.iloc[groupings[i]:groupings[i+1], 13])))
        # successes["Red. Proportional 1% MOE"][i] = 100 * np.average(np.logical_and(np.array(df_1MOE.iloc[groupings[i]:groupings[i+1], 2] < df_1MOE.iloc[groupings[i]:groupings[i+1], 19]), np.array(df_1MOE.iloc[groupings[i]:groupings[i+1], 2] > df_1MOE.iloc[groupings[i]:groupings[i+1], 18])))
        # successes["Red. Proportional 3% MOE"][i] = 100 * np.average(np.logical_and(np.array(df_3MOE.iloc[groupings[i]:groupings[i+1], 2] < df_3MOE.iloc[groupings[i]:groupings[i+1], 19]), np.array(df_3MOE.iloc[groupings[i]:groupings[i+1], 2] > df_3MOE.iloc[groupings[i]:groupings[i+1], 18])))
        # successes["Red. Proportional 5% MOE"][i] = 100 * np.average(np.logical_and(np.array(df_5MOE.iloc[groupings[i]:groupings[i+1], 2] < df_5MOE.iloc[groupings[i]:groupings[i+1], 19]), np.array(df_5MOE.iloc[groupings[i]:groupings[i+1], 2] > df_5MOE.iloc[groupings[i]:groupings[i+1], 18])))

        successesStd["Optimal 3x3"][i] = 100 * np.sqrt(successes["Optimal 3x3"][i]/100 * (1 - successes["Optimal 3x3"][i]/100)) / np.sqrt(numSamples)
        successesStd["Optimal 4x4"][i] = 100 * np.sqrt(successes["Optimal 4x4"][i]/100 * (1 - successes["Optimal 4x4"][i]/100)) / np.sqrt(numSamples)
        successesStd["Optimal 5x5"][i] = 100 * np.sqrt(successes["Optimal 5x5"][i]/100 * (1 - successes["Optimal 5x5"][i]/100)) / np.sqrt(numSamples)
        successesStd["Proportional 3x3"][i] = 100 * np.sqrt(successes["Proportional 3x3"][i]/100 * (1 - successes["Proportional 3x3"][i]/100)) / np.sqrt(numSamples)
        successesStd["Proportional 4x4"][i] = 100 * np.sqrt(successes["Proportional 4x4"][i]/100 * (1 - successes["Proportional 4x4"][i]/100)) / np.sqrt(numSamples)
        successesStd["Proportional 5x5"][i] = 100 * np.sqrt(successes["Proportional 5x5"][i]/100 * (1 - successes["Proportional 5x5"][i]/100)) / np.sqrt(numSamples)
        # successesStd["Red. Proportional 1% MOE"][i] = 100 * np.sqrt(successes["Red. Proportional 1% MOE"][i]/100 * (1 - successes["Red. Proportional 1% MOE"][i]/100)) / np.sqrt(numSamples)
        # successesStd["Red. Proportional 3% MOE"][i] = 100 * np.sqrt(successes["Red. Proportional 3% MOE"][i]/100 * (1 - successes["Red. Proportional 3% MOE"][i]/100)) / np.sqrt(numSamples)
        # successesStd["Red. Proportional 5% MOE"][i] = 100 * np.sqrt(successes["Red. Proportional 5% MOE"][i]/100 * (1 - successes["Red. Proportional 5% MOE"][i]/100)) / np.sqrt(numSamples)

        costSavingsRatio["3x3"][i] = np.average(df_3x3.iloc[groupings[i]:groupings[i+1], 5])
        costSavingsRatio["4x4"][i] = np.average(df_4x4.iloc[groupings[i]:groupings[i+1], 5])
        costSavingsRatio["5x5"][i] = np.average(df_5x5.iloc[groupings[i]:groupings[i+1], 5])

    # hatchings = [None, "||||", "////", None, "||||", "////", None, "||||", "////"]
    # colors = ["tab:blue", "tab:orange", "tab:green", "tab:blue", "tab:orange", "tab:green", "tab:blue", "tab:orange", "tab:green"]
    # markers = ["o", "o", "o", "^", "^", "^", "X", "X", "X"]
    # cnt = 0
    # for attribute, measurement in successes.items():
    #     if "Red." not in attribute:
    #         plt.errorbar(np.arange(2,numGroups+2), measurement, yerr=successesStd[attribute], c=colors[cnt])
    #         plt.plot(np.arange(2,numGroups+2), measurement,c=colors[cnt])
    #         plt.scatter(np.arange(2,numGroups+2), measurement, label=attribute, edgecolor="k", c=colors[cnt], marker=markers[cnt], hatch=hatchings[cnt],s=110)
    #     cnt += 1

    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #                   ncols=3, mode="expand", borderaxespad=0.)
    # plt.plot((-5,numGroups+5), (95,95), color="tab:gray", linestyle="dashed", alpha=0.5)
    # plt.xlim(1.5, 20.5)
    # plt.ylim(88,100)
    # plt.ylabel("Success Rate (%)")
    # plt.minorticks_on()
    # plt.show()

    fig, axs = plt.subplots(2,layout='constrained',sharex=True)

    # hatchings = [None, "||||", None, "||||", None, "||||"]
    # colors = ["tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    # markers = ["o", "o", "^", "^", "X", "X"]
    
    # cnt = 0
    # for attribute, measurement in MOEs.items():
    #     axs[0].scatter(np.arange(numGroups), measurement, label=attribute, edgecolor="k", c=colors[cnt], marker=markers[cnt], hatch=hatchings[cnt],s=125)
    #     cnt += 1
    # axs[0].plot((-5,numGroups+5), (5,5), color="tab:gray", linestyle="dashed", alpha=0.5)
    # axs[0].plot((-5,numGroups+5), (3,3), color="tab:gray", linestyle="dashed", alpha=0.5)
    # axs[0].plot((-5,numGroups+5), (1,1), color="tab:gray", linestyle="dashed", alpha=0.5)
    # axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #                   ncols=3, mode="expand", borderaxespad=0.)
    # axs[0].set_xlim(-0.5,numGroups + 0.5)
    # axs[0].set_ylabel("MOE (%)")
    # axs[0].set_yticks(np.arange(6))
    # axs[0].minorticks_on()
    
    hatchings = [None, "||||", None, "||||", None, "||||"]
    colors = ["tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    markers = ["o", "o", "^", "^", "X", "X"]
    cnt = 0
    for attribute, measurement in successes.items():
        axs[0].scatter(np.arange(numGroups), measurement, label=attribute, edgecolor="k", c=colors[cnt], marker=markers[cnt], hatch=hatchings[cnt],s=110)
        # axs[1].errorbar(np.arange(numGroups), measurement, yerr=successesStd[attribute], c=colors[cnt])
        cnt += 1
    axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=3, mode="expand", borderaxespad=0.)
    axs[0].plot((-5,numGroups+5), (95,95), color="tab:gray", linestyle="dashed", alpha=0.5)
    axs[0].set_xlim(-0.5,numGroups + 0.5)
    axs[0].set_ylim(92,100.5)
    axs[0].set_ylabel("Success Rate (%)")
    axs[0].minorticks_on()

    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "^", "X"]
    cnt=0
    for attribute, measurement in costSavingsRatio.items():
        axs[1].scatter(np.arange(numGroups), measurement, label=attribute, edgecolor="k", c=colors[cnt], marker=markers[cnt], s=125)
        cnt += 1
    axs[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=3, mode="expand", borderaxespad=0.)
    axs[1].set_xlim(-0.5,numGroups + 0.5)
    axs[1].set_ylabel("Cost Savings Ratio")
    axs[1].set_xlabel("Image Parameter Group Number")

    plt.minorticks_on()
    plt.show()

def PlotSimResults():
    # First row: MOE for all 40 groups for two allocation methods for three MOEs (scatter)
    # Second row: Percent of successes for all 40 groups for two allocation methods for three MOEs (scatter)
    # Third row: Residual (violin)
    numGroups = 40
    numSamples = 10000

    plt.rcParams["font.family"] = "serif"
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)

    MOEs = {"1% MOE": np.zeros(numGroups),
            "3% MOE" : np.zeros(numGroups),
            "5% MOE": np.zeros(numGroups)}
    
    successes = {"1% MOE": np.zeros(numGroups),
            "3% MOE" : np.zeros(numGroups),
            "5% MOE": np.zeros(numGroups)}
    
    residuals = {"1% MOE": np.zeros((numGroups,numSamples)),
            "3% MOE" : np.zeros((numGroups,numSamples)),
            "5% MOE": np.zeros((numGroups,numSamples))}

    df_1MOE = pandas.read_csv("SimOutput_95CI_1MOE_4x4.csv", header=None)
    df_3MOE = pandas.read_csv("SimOutput_95CI_3MOE_4x4.csv", header=None)
    df_5MOE = pandas.read_csv("SimOutput_95CI_5MOE_4x4.csv", header=None)

    groupings = np.arange(0,410000,10000)
    for i in range(numGroups):
        # compare 11,12,13 to 2
        MOEs["1% MOE"][i] = 100 * np.average((df_1MOE.iloc[groupings[i]:groupings[i+1], 13] - df_1MOE.iloc[groupings[i]:groupings[i+1], 12]) / 2)
        MOEs["3% MOE"][i] = 100 * np.average((df_3MOE.iloc[groupings[i]:groupings[i+1], 13] - df_3MOE.iloc[groupings[i]:groupings[i+1], 12]) / 2)
        MOEs["5% MOE"][i] = 100 * np.average((df_5MOE.iloc[groupings[i]:groupings[i+1], 13] - df_5MOE.iloc[groupings[i]:groupings[i+1], 12]) / 2)

        successes["1% MOE"][i] = 100 * np.average(np.logical_and(np.array(df_1MOE.iloc[groupings[i]:groupings[i+1], 2]/100 < df_1MOE.iloc[groupings[i]:groupings[i+1], 13]), np.array(df_1MOE.iloc[groupings[i]:groupings[i+1], 2]/100 > df_1MOE.iloc[groupings[i]:groupings[i+1], 12])))
        successes["3% MOE"][i] = 100 * np.average(np.logical_and(np.array(df_3MOE.iloc[groupings[i]:groupings[i+1], 2]/100 < df_3MOE.iloc[groupings[i]:groupings[i+1], 13]), np.array(df_3MOE.iloc[groupings[i]:groupings[i+1], 2]/100 > df_3MOE.iloc[groupings[i]:groupings[i+1], 12])))
        successes["5% MOE"][i] = 100 * np.average(np.logical_and(np.array(df_5MOE.iloc[groupings[i]:groupings[i+1], 2]/100 < df_5MOE.iloc[groupings[i]:groupings[i+1], 13]), np.array(df_5MOE.iloc[groupings[i]:groupings[i+1], 2]/100 > df_5MOE.iloc[groupings[i]:groupings[i+1], 12])))

        residuals["1% MOE"][i] = 100 * (df_1MOE.iloc[groupings[i]:groupings[i+1], 11] - df_1MOE.iloc[groupings[i]:groupings[i+1], 2]/100)
        residuals["3% MOE"][i] = 100 * (df_3MOE.iloc[groupings[i]:groupings[i+1], 11] - df_3MOE.iloc[groupings[i]:groupings[i+1], 2]/100)
        residuals["5% MOE"][i] = 100 * (df_5MOE.iloc[groupings[i]:groupings[i+1], 11] - df_5MOE.iloc[groupings[i]:groupings[i+1], 2]/100)

    fig, axs = plt.subplots(3,layout='constrained',sharex=True)
    fig.set_size_inches(13, 8)

    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "^", "X"]
    
    cnt = 0
    for attribute, measurement in MOEs.items():
        axs[0].scatter(np.arange(numGroups), measurement, label=attribute, edgecolor="k", c=colors[cnt], marker=markers[cnt],s=125)
        cnt += 1
    axs[0].plot((-5,numGroups+5), (5,5), color="tab:gray", linestyle="dashed", alpha=0.5)
    axs[0].plot((-5,numGroups+5), (3,3), color="tab:gray", linestyle="dashed", alpha=0.5)
    axs[0].plot((-5,numGroups+5), (1,1), color="tab:gray", linestyle="dashed", alpha=0.5)
    axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=3, mode="expand", borderaxespad=0.)
    axs[0].set_xlim(-0.5,numGroups + 0.5)
    axs[0].set_ylabel("MOE (%)")
    axs[0].set_yticks(np.arange(6))
    axs[0].minorticks_on()
    
    colors = ["tab:blue", "tab:orange", "tab:green"]
    markers = ["o", "^", "X"]
    cnt = 0
    for attribute, measurement in successes.items():
        axs[1].scatter(np.arange(numGroups), measurement, label=attribute, edgecolor="k", c=colors[cnt], marker=markers[cnt], s=110)
        cnt += 1
    axs[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=3, mode="expand", borderaxespad=0.)
    axs[1].plot((-5,numGroups+5), (95,95), color="tab:gray", linestyle="dashed", alpha=0.5)
    axs[1].set_xlim(-0.5,numGroups + 0.5)
    axs[1].set_ylim(92,100.5)
    axs[1].set_ylabel("Success Rate (%)")
    axs[1].minorticks_on()
    data = [residuals["1% MOE"][i,:] for i in range(residuals["1% MOE"].shape[0])]

    parts = axs[2].violinplot(data, np.arange(40), showmeans=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor("tab:green")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    
    for partname in ['cbars','cmins','cmaxes', 'cmeans']:
        parts[partname].set_edgecolor('black')
        parts[partname].set_linewidth(1)
    
    axs[2].plot((-5,numGroups+5), (0,0), color="tab:gray", linestyle="dashed", alpha=0.5)
    axs[2].set_xticks(np.arange(40))
    axs[2].set_xlim(-0.5,numGroups - 0.5)
    axs[2].set_ylabel("Residual (%)")
    axs[2].set_xlabel("Image Parameter Group Number")

    plt.minorticks_on()
    plt.savefig("myplot.png", dpi = 300)
    # plt.show()

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
    def GetImageWithGridOverlay(self, pixelRow:int, pixelCol:int, newColor:tuple, numSurroundingPixels:int, style:int) -> np.ndarray: 
        # Center
        displayImage = gray2rgb(self.originalImage)

        if style == 0:
            displayImage[pixelRow][pixelCol] = newColor

        minValue = 1 if style != 2 else 2
        
        # above
        maxVal = 3 if pixelRow > 2 else pixelRow
        for i in range(minValue, maxVal):
            displayImage[pixelRow-i][pixelCol] = newColor

        # below
        maxVal = 3 if pixelRow < self.rows-3 else self.rows-pixelRow
        for i in range(minValue, maxVal):
            displayImage[pixelRow+i][pixelCol] = newColor

        # right
        maxVal = 3 if pixelCol < self.cols-3 else self.cols-pixelCol
        for i in range(minValue, maxVal):
            displayImage[pixelRow][pixelCol+i] = newColor

        # left
        maxVal = 3 if pixelCol > 2 else pixelCol
        for i in range(minValue, maxVal):
            displayImage[pixelRow][pixelCol-i] = newColor
        
        # pad image to ensure display proper
        displayImage = np.pad(displayImage, ((numSurroundingPixels+1, numSurroundingPixels+1), (numSurroundingPixels+1, numSurroundingPixels+1), (0,0)))

        # Crop the image to center around the grid with numSurroundingPixels around
        displayImage = displayImage[pixelRow:pixelRow+2*numSurroundingPixels,pixelCol:pixelCol+2*numSurroundingPixels,:]

        return displayImage

    def GetCroppedImage(self, leftBound, rightBound, topBound, bottomBound):
        return self.originalImage[topBound:bottomBound, leftBound:rightBound]
    
    def GetCroppedAndMaskedImage(self, leftBound, rightBound, topBound, bottomBound, polygonPoints):
        return self.originalImage[topBound:bottomBound, leftBound:rightBound]

# 10000 samples of 5 images of 40 different parameters
def AnalyzeCIWidthAndCoverage():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 12

    dir = "TestCases"
    files = os.listdir(dir)
    files.sort()

    numStrata_N = 4
    MOE = 0.01
    confidence = 0.95
    numSamples = 10000 # FIXME: Change to 10000 for final run
    numImages = 5

    guessValues = np.array([0.05, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 0.95])

    designEffectHeatmapWilson = np.zeros((40,numImages), dtype=np.float32)
    coverageHeatmapWilson = np.zeros((40,numImages), dtype=np.float32)
    widthHeatmapWilson = np.zeros((40,numImages), dtype=np.float32)

    designEffectHeatmapAC = np.zeros((40,numImages), dtype=np.float32)
    coverageHeatmapAC = np.zeros((40,numImages), dtype=np.float32)
    widthHeatmapAC = np.zeros((40,numImages), dtype=np.float32)

    designEffectHeatmapCP = np.zeros((40,numImages), dtype=np.float32)
    coverageHeatmapCP = np.zeros((40,numImages), dtype=np.float32)
    widthHeatmapCP = np.zeros((40,numImages), dtype=np.float32)

    keyValues = []
    with open("key.csv", "r") as keyFile:
        reader = csv.reader(keyFile)
        for line in reader:
            temp = [line[0], line[1], line[2], line[3], line[4]]
            keyValues.append(temp)
    
    startIndex = 100000
    for i in tqdm.tqdm(range(40)):
        for j in tqdm.tqdm(range(numImages),leave=False):
            coverageSuccessesWilson = 0
            widthsWilson = []
            coverageSuccessesAC = 0
            widthsAC = []
            coverageSuccessesCP = 0
            widthsCP = []

            originalImage = io.imread(os.path.join(dir,f"{startIndex+j}.png"), as_gray=True)
            myMap = PixelMap(originalImage)

            N = myMap.numPixels
            N_h = int(N / numStrata_N**2)
            W_h = N_h/N

            initialGuesses = []

            for i2 in range(numStrata_N):
                topBound = int(i2 * myMap.rows / numStrata_N)
                bottomBound = int((i2+1) * myMap.rows/numStrata_N)
                for j2 in range(numStrata_N):
                    leftBound = int(j2 * myMap.cols / numStrata_N)
                    rightBound = int((j2+1) * myMap.cols / numStrata_N)

                    selectedArea = myMap.originalImage[leftBound:rightBound+1, topBound:bottomBound+1]

                    numPorePixels = np.sum(np.where(selectedArea != 255, 1, 0))

                    porosity = numPorePixels / ((rightBound - leftBound) * (bottomBound - topBound))

                    closestGuess = guessValues[np.argmin(np.abs(porosity - guessValues))]

                    initialGuesses.append(closestGuess)
            
            initialGuesses = np.array(initialGuesses)
            p_st = np.sum(W_h * initialGuesses)
            q_st = 1 - p_st
    
            z = scipy.stats.norm.ppf(confidence)
            
            neff_func_Wilson = lambda x: MOE - (z * np.sqrt(x) / ((x+z**2)) * (p_st * (1-p_st) + (z**2 / (4*x)))**0.5)
            neff_func_AC = lambda x: MOE - (z * np.sqrt((p_st * (1-p_st)) / (x + z**2)))
            neff_func_CP = lambda x: MOE - ((scipy.stats.binom.interval(0.95, x, p_st)[1]/x - scipy.stats.binom.interval(0.95, x, p_st)[0]/x) / 2)

            # NOTE: neff values are rounded to ensure equal allocation is possible
            neff_Wilson = scipy.optimize.fsolve(neff_func_Wilson, np.array([numStrata_N**2]))[0]
            neff_Wilson = np.ceil(neff_Wilson / numStrata_N**2) * numStrata_N**2
            neff_AC = scipy.optimize.fsolve(neff_func_AC, np.array([numStrata_N**2]))[0]
            neff_AC = np.ceil(neff_AC / numStrata_N**2) * numStrata_N**2
            # NOTE: scipy CP interval reports expected number of successes
            # NOTE: scipy fsolve does not work for this, using alternative lookup table method
            testVals = np.arange(1,10000)
            CP_return_vals = neff_func_CP(testVals)
            neff_CP = testVals[np.argmin(np.abs(CP_return_vals))]
            neff_CP = np.ceil(neff_CP/ numStrata_N**2) * numStrata_N**2

            nh_func_Wilson = lambda x: neff_Wilson - ((p_st * q_st) / np.sum((W_h**2 * initialGuesses * (1-initialGuesses)) / (x - 1)))
            nh_func_AC  = lambda x: neff_AC - ((p_st * q_st) / np.sum((W_h**2 * initialGuesses * (1-initialGuesses)) / (x - 1)))
            nh_func_CP = lambda x: neff_CP - ((p_st * q_st) / np.sum((W_h**2 * initialGuesses * (1-initialGuesses)) / (x - 1)))
            
            n_h_Wilson = scipy.optimize.fsolve(nh_func_Wilson, np.array([2]))[0]
            n_h_AC = scipy.optimize.fsolve(nh_func_AC, np.array([2]))[0]
            n_h_CP = scipy.optimize.fsolve(nh_func_CP, np.array([2]))[0]
            
            n_h_Wilson = np.ceil(n_h_Wilson)
            n_h_Wilson = np.ones(numStrata_N**2, dtype=np.int16) * int(n_h_Wilson)
            n_h_AC = np.ceil(n_h_AC)
            n_h_AC = np.ones(numStrata_N**2, dtype=np.int16) * int(n_h_AC)
            n_h_CP = np.ceil(n_h_CP)
            n_h_CP = np.ones(numStrata_N**2, dtype=np.int16) * int(n_h_CP)

            designEffectHeatmapWilson[i][j] = np.sum(n_h_Wilson) / neff_Wilson
            designEffectHeatmapAC[i][j] = np.sum(n_h_AC) / neff_AC
            designEffectHeatmapCP[i][j] = np.sum(n_h_CP) / neff_CP

            # ########################## #
            # Get pixel sample locations #
            # ########################## #
            
            for l in tqdm.tqdm(range(numSamples), leave=False):
                
                ##########
                # Wilson #
                ##########

                p_h = []
                cnt = 0
                for i2 in range(numStrata_N):
                    topBound = int(i2*myMap.rows/numStrata_N)
                    bottomBound = int((i2+1)*myMap.rows/numStrata_N)
                    for j2 in range(numStrata_N):
                        leftBound = int(j2*myMap.cols/numStrata_N)
                        rightBound = int((j2+1)*myMap.cols/numStrata_N)

                        random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_Wilson[cnt], replace=False)
                        random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                        random[0,:] += topBound
                        random[1,:] += leftBound

                        pixels = list(zip(random[0,:], random[1,:]))
                        k = 0
                        for pixel in pixels:
                            if myMap.originalImage[pixel] != 255:
                                k += 1
                        
                        p_h.append(k / len(pixels))

                        cnt += 1
                
                p_h = np.array(p_h)

                p_st = np.sum(p_h * W_h)
                x = neff_Wilson * p_st
                lowerCL = ((x + z**2/2)/(neff_Wilson+z**2)) - (z * np.sqrt(neff_Wilson) / ((neff_Wilson+z**2)) * (p_st * (1-p_st) + (z**2 / (4*neff_Wilson)))**0.5)
                upperCL = ((x + z**2/2)/(neff_Wilson+z**2)) + (z * np.sqrt(neff_Wilson) / ((neff_Wilson+z**2)) * (p_st * (1-p_st) + (z**2 / (4*neff_Wilson)))**0.5)

                if float(keyValues[10000*i + j][2])/100 < upperCL and float(keyValues[10000*i + j][2])/100 > lowerCL:
                    coverageSuccessesWilson += 1
                
                widthsWilson.append((upperCL-lowerCL)/2)

                ##########
                #   AC   #
                ##########

                p_h = []
                cnt = 0
                for i2 in range(numStrata_N):
                    topBound = int(i2*myMap.rows/numStrata_N)
                    bottomBound = int((i2+1)*myMap.rows/numStrata_N)
                    for j2 in range(numStrata_N):
                        leftBound = int(j2*myMap.cols/numStrata_N)
                        rightBound = int((j2+1)*myMap.cols/numStrata_N)

                        random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_AC[cnt], replace=False)
                        random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                        random[0,:] += topBound
                        random[1,:] += leftBound

                        pixels = list(zip(random[0,:], random[1,:]))
                        k = 0
                        for pixel in pixels:
                            if myMap.originalImage[pixel] != 255:
                                k += 1
                        
                        p_h.append(k / len(pixels))

                        cnt += 1
                
                p_h = np.array(p_h)

                p_st = np.sum(p_h * W_h)
                x = neff_AC * p_st
                lowerCL = ((x + z**2/2)/(neff_AC+z**2)) - (z * np.sqrt((p_st * (1-p_st)) / (neff_AC + z**2)))
                upperCL = ((x + z**2/2)/(neff_AC+z**2)) + (z * np.sqrt((p_st * (1-p_st)) / (neff_AC + z**2)))

                if float(keyValues[10000*i + j][2])/100 < upperCL and float(keyValues[10000*i + j][2])/100 > lowerCL:
                    coverageSuccessesAC += 1
                
                widthsAC.append((upperCL-lowerCL)/2)


                ##########
                #   CP   #
                ##########

                p_h = []
                cnt = 0
                for i2 in range(numStrata_N):
                    topBound = int(i2*myMap.rows/numStrata_N)
                    bottomBound = int((i2+1)*myMap.rows/numStrata_N)
                    for j2 in range(numStrata_N):
                        leftBound = int(j2*myMap.cols/numStrata_N)
                        rightBound = int((j2+1)*myMap.cols/numStrata_N)

                        random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_CP[cnt], replace=False)
                        random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                        random[0,:] += topBound
                        random[1,:] += leftBound

                        pixels = list(zip(random[0,:], random[1,:]))
                        k = 0
                        for pixel in pixels:
                            if myMap.originalImage[pixel] != 255:
                                k += 1
                        
                        p_h.append(k / len(pixels))

                        cnt += 1
                
                p_h = np.array(p_h)

                p_st = np.sum(p_h * W_h)
                lowerCL, upperCL = scipy.stats.binom.interval(confidence, neff_CP, p_st) # NOTE: Scipy interval returns number of successes
                lowerCL /= neff_CP
                upperCL /= neff_CP

                if float(keyValues[10000*i + j][2])/100 < upperCL and float(keyValues[10000*i + j][2])/100 > lowerCL:
                    coverageSuccessesCP += 1
                
                widthsCP.append((upperCL-lowerCL)/2)
                
            coverageHeatmapWilson[i][j] = 100 * coverageSuccessesWilson / numSamples
            widthHeatmapWilson[i][j] = 100 * np.average(widthsWilson)

            coverageHeatmapAC[i][j] = 100 * coverageSuccessesAC / numSamples
            widthHeatmapAC[i][j] = 100 * np.average(widthsAC)

            coverageHeatmapCP[i][j] = 100 * coverageSuccessesCP / numSamples
            widthHeatmapCP[i][j] = 100 * np.average(widthsCP)
        
        startIndex += 10000

    with open("Sim1output1MOE.csv", "w") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(coverageHeatmapWilson)
        writer.writerows(widthHeatmapWilson)
        writer.writerows(coverageHeatmapAC)
        writer.writerows(widthHeatmapAC)
        writer.writerows(coverageHeatmapCP)
        writer.writerows(widthHeatmapCP)

    ax = sns.heatmap(coverageHeatmapWilson,annot=True, fmt=".1f", linewidth=0.5, linecolor="k", cmap="icefire", center=95, xticklabels=np.arange(1,numImages+1))
    ax.set(xlabel="Image Number", ylabel="Image Parameter Group Number")
    plt.yticks(rotation=0)
    plt.show()

    ax = sns.heatmap(widthHeatmapWilson,annot=True, fmt=".1f", linewidth=0.5, linecolor="k", cmap="icefire", center=int(MOE*100), xticklabels=np.arange(1,numImages+1))
    ax.set(xlabel="Image Number", ylabel="Image Parameter Group Number")
    plt.yticks(rotation=0)
    plt.show()

    ax = sns.heatmap(coverageHeatmapAC,annot=True, fmt=".1f", linewidth=0.5, linecolor="k", cmap="icefire", center=95, xticklabels=np.arange(1,numImages+1))
    ax.set(xlabel="Image Number", ylabel="Image Parameter Group Number")
    plt.yticks(rotation=0)
    plt.show()

    ax = sns.heatmap(widthHeatmapAC,annot=True, fmt=".1f", linewidth=0.5, linecolor="k", cmap="icefire", center=int(MOE*100), xticklabels=np.arange(1,numImages+1))
    ax.set(xlabel="Image Number", ylabel="Image Parameter Group Number")
    plt.yticks(rotation=0)
    plt.show()

    ax = sns.heatmap(coverageHeatmapCP,annot=True, fmt=".1f", linewidth=0.5, linecolor="k", cmap="icefire", center=95, xticklabels=np.arange(1,numImages+1))
    ax.set(xlabel="Image Number", ylabel="Image Parameter Group Number")
    plt.yticks(rotation=0)
    plt.show()

    ax = sns.heatmap(widthHeatmapCP,annot=True, fmt=".1f", linewidth=0.5, linecolor="k", cmap="icefire", center=int(MOE*100), xticklabels=np.arange(1,numImages+1))
    ax.set(xlabel="Image Number", ylabel="Image Parameter Group Number")
    plt.yticks(rotation=0)
    plt.show()

def AutoAnalyzeSimImages():
    dir = "TestCases"
    files = os.listdir(dir)
    files.sort()

    numStrata_N = 4
    MOE = 0.01
    confidence = 0.95

    guessValues = np.array([0.05, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 0.95])

    useWilson = True
    useAC = True
    useCP = True

    keyValues = []
    with open("key.csv", "r") as keyFile:
        reader = csv.reader(keyFile)
        for line in reader:
            temp = [line[0], line[1], line[2], line[3], line[4]]
            keyValues.append(temp)

    with open(f"SimOutput_{int(confidence*100)}CI_{int(MOE*100)}MOE_{numStrata_N}x{numStrata_N}.csv", 'a', newline='') as csvFile:
        writer = csv.writer(csvFile)
        for iternum in tqdm.tqdm(range(len((files)))):
            originalImage = io.imread(os.path.join(dir,files[iternum]), as_gray=True)
            myMap = PixelMap(originalImage)

            N = myMap.numPixels
            N_h = int(N / numStrata_N**2)
            W_h = N_h/N

            initialGuesses = []

            for i in range(numStrata_N):
                topBound = int(i * myMap.rows / numStrata_N)
                bottomBound = int((i+1) * myMap.rows/numStrata_N)
                for j in range(numStrata_N):
                    leftBound = int(j * myMap.cols / numStrata_N)
                    rightBound = int((j+1) * myMap.cols / numStrata_N)

                    selectedArea = myMap.originalImage[leftBound:rightBound+1, topBound:bottomBound+1]

                    numPorePixels = np.sum(np.where(selectedArea != 255, 1, 0))

                    porosity = numPorePixels / ((rightBound - leftBound) * (bottomBound - topBound))

                    closestGuess = guessValues[np.argmin(np.abs(porosity - guessValues))]

                    initialGuesses.append(closestGuess)
            
            initialGuesses = np.array(initialGuesses)
            p_st = np.sum(W_h * initialGuesses)
            q_st = 1 - p_st
            z = scipy.stats.norm.ppf(confidence)
            
            neff_func_Wilson = lambda x: MOE - (z * np.sqrt(x) / ((x+z**2)) * (p_st * (1-p_st) + (z**2 / (4*x)))**0.5)
            neff_func_AC = lambda x: MOE - (z * np.sqrt((p_st * (1-p_st)) / (x + z**2)))
            neff_func_CP = lambda x: MOE - ((scipy.stats.binom.interval(0.95, x, p_st)[1]/x - scipy.stats.binom.interval(0.95, x, p_st)[0]/x) / 2)

            # NOTE: neff values are rounded to ensure equal allocation is possible
            neff_Wilson = scipy.optimize.fsolve(neff_func_Wilson, np.array([numStrata_N**2]))[0]
            neff_Wilson = np.ceil(neff_Wilson / numStrata_N**2) * numStrata_N**2
            neff_AC = scipy.optimize.fsolve(neff_func_AC, np.array([numStrata_N**2]))[0]
            neff_AC = np.ceil(neff_AC / numStrata_N**2) * numStrata_N**2
            # NOTE: scipy CP interval reports expected number of successes
            # NOTE: scipy fsolve does not work for this, using alternative lookup table method
            testVals = np.arange(1,10000)
            CP_return_vals = neff_func_CP(testVals)
            neff_CP = testVals[np.argmin(np.abs(CP_return_vals))]
            neff_CP = np.ceil(neff_CP/ numStrata_N**2) * numStrata_N**2

            nh_func_Wilson = lambda x: neff_Wilson - ((p_st * q_st) / np.sum((W_h**2 * initialGuesses * (1-initialGuesses)) / (x - 1)))
            nh_func_AC  = lambda x: neff_AC - ((p_st * q_st) / np.sum((W_h**2 * initialGuesses * (1-initialGuesses)) / (x - 1)))
            nh_func_CP = lambda x: neff_CP - ((p_st * q_st) / np.sum((W_h**2 * initialGuesses * (1-initialGuesses)) / (x - 1)))
            
            n_h_Wilson = scipy.optimize.fsolve(nh_func_Wilson, np.array([2]))[0]
            n_h_AC = scipy.optimize.fsolve(nh_func_AC, np.array([2]))[0]
            n_h_CP = scipy.optimize.fsolve(nh_func_CP, np.array([2]))[0]
            
            n_h_Wilson = np.ceil(n_h_Wilson)
            n_h_Wilson = np.ones(numStrata_N**2, dtype=np.int16) * int(n_h_Wilson)
            n_h_AC = np.ceil(n_h_AC)
            n_h_AC = np.ones(numStrata_N**2, dtype=np.int16) * int(n_h_AC)
            n_h_CP = np.ceil(n_h_CP)
            n_h_CP = np.ones(numStrata_N**2, dtype=np.int16) * int(n_h_CP)

            csvRow = keyValues[iternum]
            
            # ########################## #
            # Get pixel sample locations #
            # ########################## #
            
            ##########
            # Wilson #
            ##########

            if useWilson:

                p_h = []
                cnt = 0
                for i2 in range(numStrata_N):
                    topBound = int(i2*myMap.rows/numStrata_N)
                    bottomBound = int((i2+1)*myMap.rows/numStrata_N)
                    for j2 in range(numStrata_N):
                        leftBound = int(j2*myMap.cols/numStrata_N)
                        rightBound = int((j2+1)*myMap.cols/numStrata_N)

                        random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_Wilson[cnt], replace=False)
                        random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                        random[0,:] += topBound
                        random[1,:] += leftBound

                        pixels = list(zip(random[0,:], random[1,:]))
                        k = 0
                        for pixel in pixels:
                            if myMap.originalImage[pixel] != 255:
                                k += 1
                        
                        p_h.append(k / len(pixels))

                        cnt += 1
                
                p_h = np.array(p_h)

                p_st = np.sum(p_h * W_h)
                x = neff_Wilson * p_st
                lowerCL = ((x + z**2/2)/(neff_Wilson+z**2)) - (z * np.sqrt(neff_Wilson) / ((neff_Wilson+z**2)) * (p_st * (1-p_st) + (z**2 / (4*neff_Wilson)))**0.5)
                upperCL = ((x + z**2/2)/(neff_Wilson+z**2)) + (z * np.sqrt(neff_Wilson) / ((neff_Wilson+z**2)) * (p_st * (1-p_st) + (z**2 / (4*neff_Wilson)))**0.5)

                csvRow.extend([p_st,lowerCL,upperCL])
            
            ##########
            #   AC   #
            ##########

            if useAC:
                p_h = []
                cnt = 0
                for i2 in range(numStrata_N):
                    topBound = int(i2*myMap.rows/numStrata_N)
                    bottomBound = int((i2+1)*myMap.rows/numStrata_N)
                    for j2 in range(numStrata_N):
                        leftBound = int(j2*myMap.cols/numStrata_N)
                        rightBound = int((j2+1)*myMap.cols/numStrata_N)

                        random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_AC[cnt], replace=False)
                        random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                        random[0,:] += topBound
                        random[1,:] += leftBound

                        pixels = list(zip(random[0,:], random[1,:]))
                        k = 0
                        for pixel in pixels:
                            if myMap.originalImage[pixel] != 255:
                                k += 1
                        
                        p_h.append(k / len(pixels))

                        cnt += 1
                
                p_h = np.array(p_h)

                p_st = np.sum(p_h * W_h)
                x = neff_AC * p_st
                lowerCL = ((x + z**2/2)/(neff_AC+z**2)) - (z * np.sqrt((p_st * (1-p_st)) / (neff_AC + z**2)))
                upperCL = ((x + z**2/2)/(neff_AC+z**2)) + (z * np.sqrt((p_st * (1-p_st)) / (neff_AC + z**2)))

                csvRow.extend([p_st,lowerCL,upperCL])


            ##########
            #   CP   #
            ##########

            if useCP:
                p_h = []
                cnt = 0
                for i2 in range(numStrata_N):
                    topBound = int(i2*myMap.rows/numStrata_N)
                    bottomBound = int((i2+1)*myMap.rows/numStrata_N)
                    for j2 in range(numStrata_N):
                        leftBound = int(j2*myMap.cols/numStrata_N)
                        rightBound = int((j2+1)*myMap.cols/numStrata_N)

                        random = np.random.choice(np.arange(0,((bottomBound-topBound) * (rightBound-leftBound))), n_h_CP[cnt], replace=False)
                        random = np.array(np.unravel_index(random, (bottomBound-topBound,rightBound-leftBound)))
                        random[0,:] += topBound
                        random[1,:] += leftBound

                        pixels = list(zip(random[0,:], random[1,:]))
                        k = 0
                        for pixel in pixels:
                            if myMap.originalImage[pixel] != 255:
                                k += 1
                        
                        p_h.append(k / len(pixels))

                        cnt += 1
                
                p_h = np.array(p_h)
                p_st = np.sum(p_h * W_h)
                lowerCL, upperCL = scipy.stats.binom.interval(confidence, neff_CP, p_st) # NOTE: Scipy interval returns number of successes
                lowerCL /= neff_CP
                upperCL /= neff_CP

                csvRow.extend([p_st,lowerCL,upperCL])

            writer.writerow(csvRow)
            csvFile.flush()


# CompareTimes()
# PlotSimResultsStrata()
PlotSimResults()
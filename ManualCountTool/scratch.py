import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


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

def PlotSimResults():
    plt.rcParams["font.family"] = "serif"
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)

    porosityDict = {"5%":0, "12.5%":1, "25%":2, "37.5%":3, "50%":4}
    sizeDict = {"Small":0, "Medium":1, "Large":2, "Mixed":3}
    distributionDict = {"Random":0, "Clustered":1}

    countsPorosity = {"Optimal":[0,0,0,0,0],"Proportional":[0,0,0,0,0]}
    countsSize = {"Optimal":[0,0,0,0],"Proportional":[0,0,0,0]}
    countsDistribution = {"Optimal":[0,0],"Proportional":[0,0]}

    ratioPorosity = [[], [], [], [], []]
    ratioSize = [[], [], [], []]
    ratioDistribution = [[], []]

    with open('key.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        cnt = 0
        for row in csv_reader:
            if cnt==0:
                cnt +=1
                continue

            porosityIdx = porosityDict[row[1] + "%"]
            sizeIdx = sizeDict[row[3]]
            distIdx = distributionDict[row[4]]

            ratioPorosity[porosityIdx].append(float(row[5]) / float(row[9]))
            ratioDistribution[distIdx].append(float(row[5]) / float(row[9]))
            ratioSize[sizeIdx].append(float(row[5]) / float(row[9]))

            if float(row[7]) < float(row[2]) and float(row[2]) < float(row[8]):
                countsPorosity["Optimal"][porosityIdx] += 1
                countsSize["Optimal"][sizeIdx] += 1
                countsDistribution["Optimal"][distIdx] += 1

            if float(row[11]) < float(row[2]) and float(row[2]) < float(row[12]):
                countsPorosity["Proportional"][porosityIdx] += 1
                countsSize["Proportional"][sizeIdx] += 1
                countsDistribution["Proportional"][distIdx] += 1

    fig, ax = plt.subplots(2,3,layout='constrained')

    x = np.arange(5)  # the label locations
    width = 0.43  # the width of the bars
    multiplier = 0

    hatches = [None, "//"]

    for attribute, measurement in countsPorosity.items():
        offset = width * multiplier

        rects = ax[0][0].bar(x + offset, 100*np.array(measurement)/160, width,label=attribute, edgecolor="k",hatch=hatches[multiplier])
        ax[0][0].bar_label(rects, padding=2, fmt=lambda x: f'{x:.1f}')

        multiplier += 1
    
    ax[0][0].set_xticks(x + width/2, list(porosityDict.keys()))

    x = np.arange(4)  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    for attribute, measurement in countsSize.items():
        offset = width * multiplier

        rects = ax[0][1].bar(x + offset, 100*np.array(measurement)/200, width, label=attribute, edgecolor="k", hatch=hatches[multiplier])
        ax[0][1].bar_label(rects, padding=2, fmt=lambda x: f'{x:.1f}')

        multiplier += 1
    
    ax[0][1].set_xticks(x + width/2, list(sizeDict.keys()))

    x = np.arange(2)  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    for attribute, measurement in countsDistribution.items():
        offset = width * multiplier

        rects = ax[0][2].bar(x + offset, 100*np.array(measurement)/400, width, label=attribute, edgecolor="k", hatch=hatches[multiplier])
        ax[0][2].bar_label(rects, padding=2, fmt=lambda x: f'{x:.1f}')

        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0][0].set_ylabel('Measurements Within MOE (%)')

    ax[0][0].set_xlabel("Porosity")
    ax[0][1].set_xlabel("Pore Sizes")
    ax[0][2].set_xlabel("Pore Distribution")

    ax[0][2].set_xticks(x + width/2, list(distributionDict.keys()))
    ax[0][2].legend(bbox_to_anchor=(1.01,0.5))
    
    ax[0][0].set_ylim(0, 107)
    ax[0][1].set_ylim(0, 107)
    ax[0][2].set_ylim(0, 107)

    ax[1][0].violinplot(ratioPorosity)
    ax[1][0].set_xticks([y + 1 for y in range(len(ratioPorosity))],
                  labels=['5%', '12.5%', '25%', '37.5%', '50%'])
    ax[1][1].violinplot(ratioSize)
    ax[1][1].set_xticks([y + 1 for y in range(len(ratioSize))],
                  labels=['Small', 'Medium', 'Large', 'Mixed'])
    ax[1][2].violinplot(ratioDistribution)
    ax[1][2].set_xticks([y + 1 for y in range(len(ratioDistribution))],
                  labels=['Random', 'Clustered'])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1][0].set_ylabel('Sampling Cost Savings Ratio')
    ax[1][0].set_xlabel("Porosity")
    ax[1][1].set_xlabel("Pore Sizes")
    ax[1][2].set_xlabel("Pore Distribution")
    
    ax[1][0].set_ylim(0.85, 1.1)
    ax[1][1].set_ylim(0.85, 1.1)
    ax[1][2].set_ylim(0.85, 1.1)

    plt.show()

PlotSimResults()
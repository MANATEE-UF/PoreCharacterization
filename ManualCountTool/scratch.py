import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv

# FIXME: Instead of reducing the MOE, make it one sided
# If p=0.99 and MOE=0.05, set the upper CI to 1.0 and only care about the lower CI
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
        N=512*512
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

def TwoSidedCL_A(n, p, alpha):
    sum = 0
    A_U = 0
    while sum <= alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
        A_U += 1
        sum = 0
        for i in range(A_U):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)

    sum = 1.0
    A_L = 0
    while sum > alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
        A_L += 1
        sum = 0
        for i in range(A_L, n+1):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
    
    return A_L-1, A_U-1

def TopOneSidedCL_A(n, p, alpha):
    A_U = n

    sum = 1.0
    A_L = 0

    while sum > alpha:
        A_L += 1
        sum = 0
        for i in range(A_L, n+1):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
    
    return A_L-1, A_U

def BottomOneSidedCL_A(n, p, alpha):
    sum = 0
    A_U = 0
    while sum <= alpha:
        A_U += 1
        sum = 0
        for i in range(A_U):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)

    A_L = 0
    
    return A_L, A_U-1

def UpperCL_A(n, p, alpha):
    sum = 0
    A_U = 0
    while sum <= alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
        A_U += 1
        sum = 0
        for i in range(A_U):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
    
    return A_U-1

def LowerCL_A(n, p, alpha):
    sum = 1.0
    A_L = 0
    while sum > alpha + (1-alpha)/2: # The addition term is used because alpha includes AUC of upper and lower bound.
        A_L += 1
        sum = 0
        for i in range(A_L, n):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
    
    return A_L-1

def PlotSimResults():
    plt.rc("axes", titlesize=20)
    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=14)
    plt.rc("ytick", labelsize=14)
    categoryDict = {"random":{"small":0, "medium":1, "large":2, "mixed":3}, "clustered":{"small":4, "medium":5, "large":6, "mixed":7}}
    colorDict = {0:"deeppink", 1:"orchid", 2:"lightsteelblue", 3:"cyan", 4:"springgreen", 5:"darkkhaki", 6:"darkorange", 7:"lightcoral"}
    titleDict = {0:"Random, Small", 1:"Random, Medium", 2:"Random, Large", 3:"Random, Mixed", 4:"Clustered, Small", 5:"Clustered, Medium", 6:"Clustered, Large", 7:"Clustered, Mixed"}

    fig,axs = plt.subplots(nrows=2, ncols=4)
    with open('SimResults.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        
        # Iterate over each row in the CSV file
        cnt = 0
        for row in csv_reader:
            if cnt==0:
                cnt +=1
                continue

            idxToUse = categoryDict[row[5]][row[4]]
            unravledIdx = np.unravel_index(idxToUse, (2,4))

            axs[unravledIdx[0]][unravledIdx[1]].scatter(float(row[1]), float(row[3]), c=colorDict[idxToUse], label=f"{row[5]}, {row[4]}", edgecolors="k",s=125)    

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

PlotSimResults()
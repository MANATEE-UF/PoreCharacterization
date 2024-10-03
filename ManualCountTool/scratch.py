import numpy as np
import matplotlib.pyplot as plt
import scipy

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
        numStrata_N = 4
        N=512*512
        MOE = 0.05
        e_moe = 0.01
        alpha = 0.95
        print(f"{i+1}/21: {porosityGuess:.2f}")

        W_h = 1 / numStrata_N**2 # Value constant because image is split evenly. small differences neglected
        initialStrataProportion = porosityGuess
        variance = np.sum(W_h * np.sqrt(initialGuesses * (1 - initialGuesses)))**2 / numStrata_N**2 - ((1/N) * np.sum(W_h * initialGuesses * (1 - initialGuesses))) # highest variance given the initial guesses, assuming only one point taken per stratum
        useUpper = False
        useLower = False
        if initialStrataProportion > 0.5 and MOE > 1-initialStrataProportion:
            useUpper = True
            MOE = ((1-initialStrataProportion)+MOE) / 2 # TODO: Make decision regarding how to change MOE for upper and lower cases
            # print(f"MOE stretches beyond range of [0,1] based on initial guess, reducing to {MOE:.2f}")
        elif initialStrataProportion < 0.5 and MOE > initialStrataProportion:
            useLower = True
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
            if useUpper:
                lowerCL, upperCL = TopOneSidedCL_A(n, initialStrataProportion, alpha)
                lowerCL /= n
                upperCL /= n
            elif useLower:
                lowerCL, upperCL = BottomOneSidedCL_A(n, initialStrataProportion, alpha)
                lowerCL /= n
                upperCL /= n
            else:
                lowerCL, upperCL = TwoSidedCL_A(n, initialStrataProportion, alpha)
                lowerCL /= n
                upperCL /= n

            if ((upperCL - lowerCL) / 2) > MOE: # Eq.15 not satisfied
                variance *= d
            else: # Eq. 15 satisfied
                pctDiff = abs((((upperCL - lowerCL) / 2) - MOE) / MOE)
                if pctDiff > e_moe: # variance too low, overestimating how many sample points needed
                    variance /= d
                    d += (1-d)/2
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
        for i in range(A_L, n):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
    
    return A_L, A_U

def TopOneSidedCL_A(n, p, alpha):
    A_U = n

    sum = 1.0
    A_L = 0

    while sum > alpha:
        A_L += 1
        sum = 0
        for i in range(A_L, n):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)
    
    return A_L, A_U

def BottomOneSidedCL_A(n, p, alpha):
    sum = 0
    A_U = 0
    while sum <= alpha:
        A_U += 1
        sum = 0
        for i in range(A_U):
            sum += scipy.special.comb(n, i, exact=True) * np.power(p,i) * np.power((1-p),n-i)

    A_L = 0
    
    return A_L, A_U

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

PlotSamplesPerInitialGuess()
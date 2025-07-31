import scipy.stats as st
import scipy
import numpy as np
import os
import matplotlib.pyplot as plt


def computeProbabilityDistribution(data: np.matrix, disType: str = "norm"):
    #data NxM where N is the number of features, M the number of observations

    distributions=[]
    disparams = []
    disTypes = []
    scores = []

    ncomp = data.shape[0]
    nobs=data.shape[1]
    if disType is None or disType == "auto":
        for i in range(ncomp):
            y = data[:, i]
            type, p, params = getBestProbabilityDistribution(y)
            disparams.append(params)
            disTypes.append(type)
            scores.append(p)
    else:
        dis = getattr(st, disType)
        for i in range(ncomp):
            y = data[:, i]
            params = dis.fit(y)

            disparams.append(params)
            disTypes.append(disType)
            D, p = st.kstest(y, disType, args=params)
            scores.append(p)

    x = np.arange(nobs)
    for i in range(ncomp):
        dist: st.norm_gen = getattr(st, disTypes[i])
        params=disparams[i]

        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        if arg:
            pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale) * nobs
        else:
            pdf_fitted = dist.pdf(x, loc=loc, scale=scale) * nobs
        distributions.append((x,pdf_fitted))

    return distributions, disparams,scores,disTypes

def plotProbabilityDistribution(foldername:str, distributions):
    #data NxM where N is the number of features, M the number of observations
    assert os.path.exists(foldername)

    fig=plt.figure()
    for i in range(len(distributions)):
        fig.clear()
        y = scipy.stats.vonmises.rvs(4.99,size=3000)
        h = plt.hist(y, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        
        x,pdf_y=distributions[i]
        plt.plot(x,pdf_y)
        plt.xlim(0,47)
    
        fig.savefig(os.path.join(foldername,f"feature_{str(i)}.svg"))
        fig.clear()
    plt.close()


def getBestProbabilityDistribution(data):
    dist_names = [
        "norm", "exponweib", "weibull_max", "weibull_min", "pareto",
        "genextreme"
    ]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for " + dist_name + " = " + str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: " + str(best_dist))
    print("Best p value: " + str(best_p))
    print("Parameters for the best fit: " + str(params[best_dist]))

    return best_dist, best_p, params[best_dist]
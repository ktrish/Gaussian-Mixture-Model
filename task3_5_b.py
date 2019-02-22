import numpy as np
import matplotlib.pyplot
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

UBIT = "tkaushik"
np.random.seed(sum([ord(c) for c in UBIT]))

#initialising the datapoints
X = np.genfromtxt('faithful.csv', dtype=float, delimiter=',')
k=3
N=272
#Initialising the Mu Sigma and theweights
Mu1=[4.0, 81]
Mu1=np.array(Mu1).reshape(2,)
Mu2=[2.0, 57]
Mu2=np.array(Mu2).reshape(2,)
Mu3=[4.0, 71]
Mu3=np.array(Mu3).reshape(2,)
sigma1=[[1.30,13.98],[13.98,184.82]]
sigma1=np.array(sigma1)
sigma2=sigma1
sigma3=sigma1
phi1=1/3
phi2=1/3
phi3=1/3

def GMM(X,Mu1,Mu2,Mu3,sigma1,sigma2,sigma3,phi1,phi2,phi3):
    #Calculating the probability density function
    pdf1 = (phi1)*multivariate_normal.pdf(X,Mu1,sigma1)
    pdf2 = (phi2)*multivariate_normal.pdf(X,Mu2,sigma2)
    pdf3 = (phi3)*multivariate_normal.pdf(X,Mu3,sigma3)
    pdf=np.column_stack((pdf1,pdf2,pdf3))
    denominator=pdf.sum(axis=1)
    for i in range(len(pdf)):
        for j in range(len(pdf[0])):
            pdf[i][j]=pdf[i][j]/denominator[i]
    W=pdf
    W_j=W.sum(axis=0)
    ######Updating the Means#################
    n_Mu1=list()
    for i in range(len(X)):
        n_Mu1.append(W[i][0]*X[i])
    n_Mu1=np.array(n_Mu1)
    n_Mu1=n_Mu1.sum(axis=0)
    n_Mu1=np.divide(n_Mu1,W_j[0])
    n_Mu2=list()
    for i in range(len(X)):
        n_Mu2.append(W[i][1]*X[i])
    n_Mu2=np.array(n_Mu2)
    n_Mu2=n_Mu2.sum(axis=0)
    n_Mu2=np.divide(n_Mu2,W_j[1])
    n_Mu3=list()
    for i in range(len(X)):
        n_Mu3.append(W[i][2]*X[i])
    n_Mu3=np.array(n_Mu3)
    n_Mu3=n_Mu3.sum(axis=0)
    n_Mu3=np.divide(n_Mu3,W_j[2])
    #####Updating Phi#######
    phi1=np.divide(W_j[0],N)
    phi1=np.divide(W_j[1],N)
    phi1=np.divide(W_j[2],N)

    #####Updating the Variance#######
    sig1=[]
    for i in range(N):
        diff=(X[i]-n_Mu1).reshape(2,1)
        diff_t=np.transpose(diff)
        res=np.dot(W[i][0],np.dot(diff,diff_t))
        sig1.append(res)
    sig1=np.divide(np.sum(sig1,axis=0),W_j[0])
    sig2=[]
    for i in range(N):
        diff=(X[i]-n_Mu2).reshape(2,1)
        diff_t=np.transpose(diff)
        res=np.dot(W[i][1],np.dot(diff,diff_t))
        sig2.append(res)
    sig2=np.divide(np.sum(sig2,axis=0),W_j[1])
    sig3=[]
    for i in range(N):
        diff=(X[i]-n_Mu3).reshape(2,1)
        diff_t=np.transpose(diff)
        res=np.dot(W[i][2],np.dot(diff,diff_t))
        sig3.append(res)
    sig3=np.divide(np.sum(sig3,axis=0),W_j[2])
    return pdf1,pdf2,pdf3,n_Mu1,n_Mu2,n_Mu3,sig1,sig2,sig3,phi1,phi2,phi3

###github reference for plotting the ellipse for each cluster
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip

for i in range(5):
    pdf1,pdf2,pdf3,Mu1,Mu2,Mu3,sigma1,sigma2,sigma3,phi1,phi2,phi3=GMM(X,Mu1,Mu2,Mu3,sigma1,sigma2,sigma3,phi1,phi2,phi3)
    x, y = X.T
    plt.plot(x, y, 'ro')
    #Plotting each of the clusters for each iteration
    plot_cov_ellipse(sigma1, Mu1, nstd=2, ax=None, color='red')
    plot_cov_ellipse(sigma2, Mu2, nstd=2, ax=None, color='green')
    plot_cov_ellipse(sigma3, Mu3, nstd=2, ax=None, color='blue')
    plt.savefig('task3_gmm_iter'+str(i+1)+'.jpg')
    plt.close(1)

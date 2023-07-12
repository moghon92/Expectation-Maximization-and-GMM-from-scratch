import os
import scipy.io as sio
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal as mvn

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

np.random.seed(3)


def read_data():
    data_file_path = os.getcwd() + '\\data\\data.mat'
    label_file_path = os.getcwd() + '\\data\\label.mat'

    data_matFile = sio.loadmat(data_file_path)
    label_matFile = sio.loadmat(label_file_path)

    data = data_matFile['data'].T
    label = label_matFile['trueLabel']

    return data, label




def GMM(data, plot=False):

    m, n = data.shape
    data_mean = np.mean(data)

    ####################### PCA #########################
    d = 4  # reduced dimension

    ndata = preprocessing.scale(data)   # scale the data
    C = np.matmul(ndata.T, ndata)/m # covariance matrix
    U,Sigma,V = np.linalg.svd(C)  # SVD

    U = U[:, :d]  # get 1st PCs
    Sigma = np.diag(Sigma[:d])    # get 1st PCs
    pdata = np.dot(ndata, U)  # project the data to the top 2 principal directions


    ##################### GMM and EM #####################
    K = 2

    # initial mean (2x4)
    mu = np.random.randn(K, d)
    #mu_old = mu.copy()

    # initial PSD covariance matrix (4x4)
    cov = []
    S1 = np.random.randn(d, d)  # generate two Gaussian random matrix
    S2 = np.random.randn(d, d)  # generate two Gaussian random matrix
    cov.append(S1@S1.T + np.identity(d))  # sigma1
    cov.append(S2@S2.T + np.identity(d))  # sigma2

    # initialize prior
    pi = np.random.random(K)
    pi = pi/np.sum(pi)

    # initialize the posterior
    tau = np.full((m, K), fill_value=0.)

    #plt.ion()

    maxIter= 100
    tol = 1e-6

    log_like = []
    ll_old = 0.0
    for i in range(maxIter):

        # E-step
        for k in range(K):
            tau[:, k] = pi[k] * mvn.pdf(pdata, mu[k], cov[k])
        # normalize tau
        sum_tau = np.sum(tau, axis=1)
        sum_tau.shape = (m,1)
        tau = np.divide(tau, np.tile(sum_tau, (1, K)))

        # M-step
        for kk in range(K):
            # update prior
            pi[kk] = np.sum(tau[:, kk])/m

            # update component mean
            mu[kk] = pdata.T @ tau[:,kk] / np.sum(tau[:,kk], axis = 0)

            # update cov matrix
            dummy = pdata - np.tile(mu[kk], (m,1)) # X-mu
            cov[kk] = dummy.T @ np.diag(tau[:,kk]) @ dummy / np.sum(tau[:,kk], axis = 0)

        print('-----iteration---',i)

        ll = np.full((m, K), fill_value=0.)
        for kkk in range(K):
            ll[:,kkk] = pi[kkk] * mvn.pdf(pdata, mu[kkk], cov[kkk])

        ll = np.log(np.sum(ll) )
        log_like.append( ll )


        if np.linalg.norm(ll-ll_old) < tol:
            print('training coverged')
            break
        ll_old = ll.copy()

    preds = np.argmax(tau, axis=1)

    for i in range(K):
        reconst_mean = U@(Sigma**0.5)@mu[i] + data_mean
        #reconst_cov = U@(Sigma**0.5)@cov[i]@(Sigma**0.5)@U.T
        fig, ax = plt.subplots()
        plt.imshow(reconst_mean.reshape(28,28).T, cmap='gray')#, vmin=0, vmax=255)
        plt.savefig(f'avg_image_comp_{i+1}.png')


        fig0, ax0 = plt.subplots()
        sns.heatmap(cov[i], ax=ax0)
        plt.title('Cov_matrix')
        plt.savefig(f'Cov_matrix_comp_{i+1}.png')

    fig1, ax1 = plt.subplots()
    ax1.plot(log_like)
    plt.xlabel('iteration')
    plt.ylabel('ll')
    plt.title('log-likelihood')
    plt.savefig('log-likelihood.png')

    fig2, ax2 = plt.subplots()
    ax2.scatter(pdata[:,0], pdata[:,1], c=tau[:,0])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('GMM.png')

    if plot:
        plt.show()

    return preds


def main():
    data, label = read_data()
    gmm_pred = GMM(data, plot=True)


    gmm_pred[gmm_pred == 0]=2
    gmm_pred[gmm_pred == 1]=6

    _, class_totals = np.unique(label, return_counts=True)
    _, miss_class = np.unique(label[label != gmm_pred], return_counts=True)
    print('GMM mis-classification rate for class 2:', miss_class[0]/class_totals[0])
    print('GMM mis-classification rate for class 6:', miss_class[1]/class_totals[1])
    print('GMM mis-classification rate: ', np.sum(label != gmm_pred)/label.shape[1])
    print()

    kmeans = KMeans(n_clusters=2).fit(data)
    kmeans_preds = kmeans.labels_

    kmeans_preds[kmeans_preds == 0] = 2
    kmeans_preds[kmeans_preds == 1] = 6

    _, miss_class = np.unique(label[label != kmeans_preds], return_counts=True)
    print('KMEANS mis-classification rate for class 2:', miss_class[0]/class_totals[0])
    print('KMEANS mis-classification rate for class 6:', miss_class[1]/class_totals[1])
    print('KMEANS mis-classification rate: ', np.sum(label != kmeans_preds)/label.shape[1])

if __name__ == "__main__":
    main()
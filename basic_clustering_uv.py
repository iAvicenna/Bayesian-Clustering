#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:06:50 2023

@author: avicenna
"""

import pymc as pm
import numpy as np
import arviz as az
import pytensor.tensor as ptt
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
from sklearn import mixture


def sample_cluster_likelihoods(model, nclusters_fit, data, trace):
    '''
    sampling the cluster likelihoods from a given model and trace
    '''

    with model:

        μ = model.μ
        σ = model.σ
        w = model.w

        components = [pm.Normal.dist(μ[i,:], σ) for i in range(nclusters_fit)]

        log_p =\
            ptt.stack([pm.logp(components[i],data).sum(axis=1)
                       for i in range(nclusters_fit)])

        p = pm.math.exp(log_p)

        normalization = (w[:,None]*p).sum(axis=0)

        pm.Deterministic("cluster_likelihoods",
                         w[:,None]/normalization[None,:]*p)


        pps = pm.sample_posterior_predictive(trace,
                                             var_names=["cluster_likelihoods"])

    return pps


def compute_cluster_likelihoods(data, trace=None, MAP=None):

    '''
    computing the cluster likelihoods from a given model
    and trace/MAP
    '''

    ndims = data.shape[1]

    if trace is not None:
        u = az.summary(trace, var_names="μ")
        w = az.summary(trace, var_names="w").iloc[:,0].values
        σ = az.summary(trace, var_names="σ").iloc[:,0].values

        mus = np.reshape(u.iloc[:,0].values, (w.size, ndims))

    elif MAP is not None:
        mus = MAP["μ"]
        w = MAP["w"]
        σ = MAP["σ"]


    max_nclusters = w.size

    components = [pm.Normal.dist(mus[i,:],    σ) for i
                                in range(max_nclusters)]

    log_p =\
        np.array([np.sum(pm.logp(components[i],data).eval(),axis=1)
                  for i in range(max_nclusters)])
    p = np.exp(log_p)

    normalization = np.sum(w[:,None]*p, axis=0)

    cluster_likelihoods = w[:,None]/normalization[None,:]*p


    return cluster_likelihoods


def generate_data(ndims, nclusters, max_len, sd, ndata, seed=0):

    '''
    generate simple trial data for clustering
    '''

    if ndims==2:
        fig,ax = plt.subplots(1,1, figsize=(5,5))

    rng = np.random.default_rng(seed)

    centers = rng.random((nclusters,ndims))*2*max_len - max_len

    centers = centers[np.argsort(centers[:,0]),:]

    ndata_per_cluster = rng.poisson(ndata, size=nclusters)

    data = np.zeros((np.sum(ndata_per_cluster), ndims))
    labels = np.zeros((np.sum(ndata_per_cluster),))

    for i in range(nclusters):
        s0 = np.sum(ndata_per_cluster[:i])
        s1 = s0 + ndata_per_cluster[i]

        labels[s0:s1] = i

        data[s0:s1,:] = centers[i,:][None,:] +\
            rng.normal(0, sd, size=(ndata_per_cluster[i], ndims))

        if ndims==2:
            ax.scatter(centers[i,0], centers[i,1], color="red", zorder=1)
            ax.scatter(data[s0:s1,0], data[s0:s1,1], color="black", alpha=0.1,
                       zorder=0)

    return data, centers, labels



def init_vals(nclusters, data):
    '''
    predict cluster centers using spectral clustering
    '''

    ndims = data.shape[1]

    clustering = SpectralClustering(n_clusters=nclusters,
                                    assign_labels='discretize',
                                    random_state=0).fit(data)

    labels_sc = clustering.labels_.astype(int)
    centers_sc = np.zeros((nclusters,ndims))

    for i in range(nclusters):
        centers_sc[i,:] = data[labels_sc==i,:].mean(axis=0)

    centers_sc = centers_sc[np.argsort(centers_sc[:,0]),:]

    return centers_sc, labels_sc



def plot(cluster_likelihoods, colors, centers, data,
         dim0=0, dim1=1):

    '''
    plot the result of clustering projected to given dims dim0, dim1
    '''

    max_len = np.max(np.linalg.norm(data-data.mean(axis=0), axis=1))

    nclusters = centers.shape[0]
    ndata,_ = data.shape
    fig,ax = plt.subplots(1,1, figsize=(5,5))

    for i in range(nclusters):
        ax.scatter(centers[i,dim0],centers[i,dim1], color="white", zorder=1,
                   marker="^", edgecolor="black", s=100, linewidth=2)


    for i in range(ndata):

        l = cluster_likelihoods[:,i][:,None]
        color = np.sum(l*colors,axis=0)
        ax.scatter(data[i,dim0], data[i,dim1], color=color, zorder=0,
                   edgecolor="black")
        ax.grid("on", alpha=0.2)

    ax.set_xlim(-max_len, max_len)
    ax.set_ylim(-max_len, max_len)

    return fig,ax


def bayesian_clustering(data, nclusters_fit, conc=1, mu_sigma=10, alpha=2,
                        beta=1, est_centers=None, sample=False):

    if est_centers is None:
        init0 = np.linspace(-np.max(np.abs(data)), np.max(np.abs(data)),
                            nclusters_fit)
        init1 = None
        center_mus = [data.mean(axis=0)[0],  data.mean(axis=0)[1:]]
    else:
        init0 = est_centers[:,0]
        init1 = est_centers[:,1:]
        center_mus = [est_centers[:,0], est_centers[:,1:]]

    ndims = data.shape[1]

    with pm.Model() as model:

        #priors
        μ0 = pm.Normal("μ0",
                       mu=center_mus[0],
                       sigma=mu_sigma,
                       shape=(nclusters_fit,),
                       transform=pm.distributions.transforms.univariate_ordered,
                       initval=init0)

        μ1 = pm.Normal("μ1",
                       mu=center_mus[1],
                       sigma=mu_sigma,
                       shape=(nclusters_fit, ndims-1),
                       initval=init1)

        σ = pm.InverseGamma("σ", alpha=alpha, beta=beta)
        weights = pm.Dirichlet("w", conc*np.ones(nclusters_fit))

        #transformed priors
        μ = pm.Deterministic("μ", ptt.concatenate([μ0[:,None], μ1], axis=1))
        components = [pm.Normal.dist(μ[i,:], σ) for i in range(nclusters_fit)]

        #likelihood
        pm.Mixture('like', w=weights, comp_dists=components, observed=data)

        if sample:
            trace = pm.sample(draws=4000, chains=6, tune=2000,
                              target_accept=0.95)
        else:
            MAP = pm.find_MAP()

    if sample:
        return trace, model

    return MAP, model


def main():

    ndims = 2
    nclusters = 4
    max_len = 5
    seed = 0
    sd = 1
    ndata = 50

    data, _, labels_true = generate_data(ndims, nclusters, max_len,
                                         sd, ndata, seed=seed)
    nclusters_fit = nclusters

    centers_sc, labels_sc = init_vals(nclusters_fit, data)

    gmm = mixture.GaussianMixture(n_components=nclusters_fit).fit(data)
    labels_sk = gmm.predict(data)

    sample = True

    if not sample:
        MAP, model = bayesian_clustering(data, nclusters_fit, est_centers=centers_sc,
                                         sample=sample)
        mus = MAP["μ"]
        cluster_likelihoods = compute_cluster_likelihoods(data, model, MAP=MAP)

    else:
        trace, model = bayesian_clustering(data, nclusters_fit, est_centers=centers_sc,
                                           sample=sample, mu_sigma=5)
        mus = np.reshape(az.summary(trace, var_names="μ").iloc[:,0].values,
                         (nclusters, ndims))
        pps = sample_cluster_likelihoods(model, nclusters_fit, data, trace)

        cluster_likelihoods = az.summary(pps, var_names=["cluster_likelihoods"])
        cluster_likelihoods =\
            np.reshape(cluster_likelihoods.iloc[:,0].values,
                       (nclusters_fit, data.shape[0]))

    labels_fit = np.argmax(cluster_likelihoods, axis=0)

    colors = np.array([[1,0,0],[0,0,1],[0,1,0],[1,1,0], [0,0,0],
                       [1,1,1]])[:nclusters_fit,:]

    plot(cluster_likelihoods, colors, mus, data)

    print("spectral clustering confusion matrix")
    print(confusion_matrix(labels_true, labels_sc))

    print("pymc bayesian clustering confusion matrix")
    print(confusion_matrix(labels_true, labels_fit))

    print("sklearn bayesian clustering confusion matrix")
    print(confusion_matrix(labels_true, labels_sk))


if __name__ == "__main__":

    main()

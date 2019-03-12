# -*- coding: utf-8 -*-
"""
Class implementeation of singular spectrum analysis (SSA) inspired by

    https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

Authors: Sebastian Glane and Rodrigo Chi
"""
import numpy as np
import pandas as pd

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def block_hankel(shape, blocks):
    """
    Computes a Hankel block matrix X, such that
    
        +----------+------------+----------+------------+
        | block[0] | block[1]   |    ...   | block[n]   |
        +----------+------------+----------+------------+
        | block[1] | block[2]   |    ...   | block[n+1] |
        +----------+------------+----------+------------+
        | block[2] | block[3]   |    ...   | block[n+2] |
        +----------+------------+----------+------------+
        |    ...   |    ...     |    ...   |    ...     |
        +----------+------------+----------+------------+
        | block[m] | block[m+1] |    ...   | block[m+n] |
        +----------+------------+----------+------------+
        
    It is assumed that all the blocks are Hankel matrices, such that the result
    is also a Hankel matrix.
    
    Parameters
    ----------
    shape : list, tuple
        Specifies the shape of Hankel block matrix.

    blocks : list, tuple
        List containing blocks to be distributed along the antidiagonals.
        block[0] occupies the 0th antidiagonal, block[1] the 1st antidiagonal,
        etc.

    Returns
    ----------
    H : numpy.ndarray
        Block Hankel matrix.
    """
    assert isinstance(shape, (tuple, list))
    assert len(shape) == 2
    m, n = shape
    assert m > 0 and n > 0
    
    assert isinstance(blocks, (tuple, list))
    assert all(isinstance(b, np.ndarray) for b in blocks)
    assert len(blocks) == m + n - 1
    
    block_array =  [[None for j in range(n)] for i in range(m)]

    # there will be m + n - 1 diagonals
    for d in range(m + n): 
        # get column index of the first element in this diagonal
        # the index is 0 for the first m lines and d - m for the remaining lines.
        start_col = max(0, d + 1 - m)
  
        # get count of elements in this diagonal
        count = min(d + 1, (n - start_col), m) 
  
        # set elements of this antidiagonal
        for j in range(count):
            block_array[min(m, d + 1) - j - 1][start_col + j] = blocks[d]

    return np.block(block_array)

def elementary_embedding(x, l):
    """
    Computes elementary embedding of a vector ``X`` with window length ``l``.
    The result is a Hankel matrix.
    
    Parameters
    ----------
    x : numpy.ndarray
        Vector to embed.
    l : int
        Window length.
    
    Returns
    ----------
    X : numpy.ndarray
        Embedding of x, a Hankel matrix.
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(l, int)
    assert x.ndim == 1
    n = x.size
    assert l > 0 and l <= n / 2
    k = n - l + 1
    from scipy.linalg import hankel
    return hankel(x, np.zeros(l))[:k,:].T

def vectorize(X):
    """
    Vectorization of 2D-array X in column-major order.
    
    Parameters
    ----------
    X : numpy.ndarray
        2D-array X.
    
    Returns
    ----------
    x : numpy.ndarray
        Vectorization of X.
    """
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    return X.flatten(order="F")

def devectorize(x, m):
    """
    Devectorization of vector x to a m-by-n array X in column-major order.
    
    Parameters
    ----------
    x : numpy.ndarray
        Vector x.

    m : int
        Integer specifying number of rows.
    
    Returns
    ----------
    X : numpy.ndarray
        Devectorization of X.
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim == 1
    assert isinstance(m, int)
    assert x.size % m == 0
    return x.reshape((m, x.size / m), order="F")

def projection_matrix(Nx, Ny, L=None, index_set=None):
    """
    Computes a projection matrix for the given index set.
    """
    assert isinstance(Nx, int) and Nx > 0
    assert isinstance(Ny, int) and Ny > 0
    m = Nx * Ny
    n = -1
    
    # index set mode, required for shaped SSA
    if index_set is not None:
        assert isinstance(index_set, set)
        assert all(isinstance(index, (tuple, list)) \
                        and len(index) == 2 for index in index_set)
        
        # size of domain space
        n = len(index_set)
        
        # lexicographical order of index_set
        indices = np.array(list(index_set))
        if indices.shape[0] != 2:
            assert indices.shape[1] == 2
            indices = indices.T
        indices = indices[:,np.lexsort(indices)]
        
        # array of row and column indices
        row_indices = np.empty((n, ), dtype=np.int)
        col_indices = np.arange(n, dtype=np.int)
        
        # ravel row the index
        for i, index in enumerate(indices.T):
            assert index[0] < Nx
            assert index[1] < Ny
            row_indices[i] = np.ravel_multi_index(index, (Nx, Ny), order="F")

    # windows size mode for rectangular data sets    
    elif L is not None:
        assert isinstance(L, (tuple, list)) and len(L) == 2
        Lx, Ly = L
        assert isinstance(Lx, int) and isinstance(Ly, int)
        
        # size of domain space
        n = Lx * Ly
        
        # array of row and column indices
        row_indices = np.empty((n, ), dtype=np.int)
        col_indices = np.arange(n, dtype=np.int)

        # ravel row the index for pairwise indices
        import itertools
        for i, index in enumerate(itertools.product(xrange(Lx), xrange(Ly))):
            row_indices[i] = np.ravel_multi_index(index, (Nx, Ny), order="F")
    else:
        raise ValueError()
        
    assert n > 0

    # initialize array of zeros
    P = np.zeros((m, n))
    
    # set all indices at once
    P[row_indices, col_indices] = 1.0
    return P

def diagsums(U, V, N, L):
    """
    Diagonal summation algorithm.
    """
    assert isinstance(U, np.ndarray)
    assert isinstance(V, np.ndarray)
    assert U.ndim == 1
    assert V.ndim == 1
    l = U.size
    k = V.size
    
    assert isinstance(N, (tuple, list)) and len(N) == 2
    Nx, Ny = N
    assert isinstance(Nx, int) and isinstance(Ny, int)
    
    assert isinstance(L, (tuple, list)) and len(L) == 2
    Lx, Ly = L
    assert isinstance(Lx, int) and isinstance(Ly, int)
    assert Lx < Nx and Ly < Ny
    assert Lx * Ly == l
    
    Kx, Ky = Nx - Lx + 1, Ny - Ly + 1
    K = (Kx, Ky)
    assert Kx * Ky == k
  
    # projection and devectorization
    Pl = projection_matrix(Nx, Ny, L=L)
    UU = devectorize(Pl.dot(U), Nx)
    # projection and devectorization    
    Pk = projection_matrix(Nx, Ny, L=K)
    VV = devectorize(Pk.dot(V), Nx)
    assert UU.shape == N
    assert VV.shape == N
    
    from scipy.fftpack import fft2, ifft2
    return ifft2(fft2(UU) * fft2(VV)).real
    

def compute_spectrum(X, fs):
    """
    Computes the fourier of multiple equidistantly samples signals at a time.
    
    Parameters
    ----------
    X : numpy.ndarray
        Array of signals where ``X[:,k]`` is the k-th signal.
    fs : float
        Sampling frequency.
    
    Returns
    ----------
    f : numpy.ndarray
        Freqeuency array.
    P : numpy.ndarray
        Power spectrum of the signals.
    """
    assert isinstance(X, np.ndarray)
    assert isinstance(fs, float) and fs > 0.
    assert X.ndim <= 2
    
    from scipy.fftpack import fft, fftfreq
    from scipy.signal import blackman
    n = X.shape[0]
    f = fftfreq(n, d=1. / fs)
    w = blackman(n)
    spec = fft(( (X - X.mean(axis=0) ).T * w ).T, axis=0)

    return np.abs(spec[:n//2]), f[:n//2]

def hankelize(X):
    """
    Hankelizes a matrix X by performing diagonal averaging.
    
    Parameters
    ----------
    X : numpy.ndarray
        Matrix X which is hankelized.
        
    Returns
    ----------
    H : numpy.ndarray
        Hankelization of matrix X.
    """
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    
    c = np.array([X[::-1].diagonal(i).mean() for i in range(-X.shape[0] + 1,  X.shape[1])])
    
    from scipy.linalg import hankel        
    return hankel(c)[:X.shape[0],:X.shape[1]]
    
def rename_columns(data_frame, group_labels=False):
    assert isinstance(data_frame, pd.DataFrame)
    assert isinstance(group_labels, bool)
    if group_labels:
        str_fun = lambda i: r"$G_{" + "{}".format(i) + "}$"
    else:
        str_fun = lambda i: r"$F_{" + "{}".format(i) + "}$"
    
    data_frame.rename(columns={i: str_fun(i) for i in data_frame.columns},
                      inplace=True)

class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    __tol = 1e-12
    __relative_ratio = 1e-2
    __power_threshold = 1e-3
    
    def __init__(self, data, l, save_mem=True, svd_method="svd"):
        """
        Decomposes the given data using singular-spectrum analysis. 
        Assumes the values of the time series are recorded at equal intervals.
        
        Parameters
        ----------
        tseries : pandas.Series, numpy.ndarray, list
            The original time series, in the form of a pandas Series, numpy array or list. 
        l : int
            The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : bool
            Conserve memory by not retaining the elementary matrices. Recommended for long time series with thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be in the form of a Pandas Series or DataFrame object.
        """
        
        # tedious type-checking for the initial time series
        if not isinstance(data, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        assert isinstance(save_mem, bool)
        
        self._ndim = data.ndim
        
        if self._ndim == 1:
            self._n = data.size
            
            self._orig_ts = pd.Series(data)

            # check window length input
            assert isinstance(l, int) and l > 0
            if not 2 <= l <= self._n / 2:
                raise ValueError("The window length must be in the interval [2, N/2].")
            self._l = l
            
            self._k = self._n - self._l + 1
        else:
            self._n = data.shape

            self._orig_ts = pd.DataFrame(data)
            
            # check window length input
            assert isinstance(l, (tuple, list)) and len(l) == data.ndim
            for i in range(data.ndim):
                if not 2 <= l[i] <= self._n[i] / 2:
                    raise ValueError("The window length must be in the interval [2, N/2].")
            self._l = l
            
            self._k = []
            for i in range(data.ndim):
                self._k.append(self._n[i] - self._l[i] + 1)

        # embedding step
        self._embed()
        
        # decomposing step
        self._compute_svd(method=svd_method)
        self._reconstruct()
        
        # grouping step
        self._compute_correlation()
        _, cum_power, _ = self._compute_powers()
        cutoff_index = np.where(cum_power > 1.0 - self.__power_threshold)[0][0]
        self._group(cutoff_index=cutoff_index)
    
    def _compute_components(self, save_mem=True, max_index=None):
        """
        Computes the elementary components of the data from the SVD.
        """
        print "Computing components..."
        assert isinstance(save_mem, bool)
        assert hasattr(self, "_d")
        assert hasattr(self, "_s")
        assert hasattr(self, "_U")
        assert hasattr(self, "_VT")
        self._ts_components = np.zeros((self._n, self._d))
        
        if not save_mem:
            # construct and save all the elementary matrices
            if max_index is None:
                self._X_elem = np.array([self._s[i] * np.outer(self._U[:,i], self._VT[i,:]) for i in range(self._d)])

                for i in range(self._d):
                    # flip rows
                    X_rev = self._X_elem[i,::-1]
                    # compute diagonal average and store them as columns
                    self._ts_components[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            else:
                assert max_index <= self._d
                
                self._X_elem = np.array([self._s[i] * np.outer(self._U[:,i], self._VT[i,:]) for i in range(max_index)])
                
                for i in range(max_index):
                    # flip rows
                    X_rev = self._X_elem[i,::-1]
                    # compute diagonal average and store them as columns
                    self._ts_components[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
                
                for i in range(max_index, self._d):
                    # compute elementary matrix
                    X_elem = self._s[i] * np.outer(self._U[:,i], self._VT[i,:])
                    # flip rows
                    X_rev = X_elem[::-1]
                    # compute diagonal average and store them as columns
                    self._ts_components[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
        else:
            # reconstruct the elementary matrices without storing them
            for i in range(self._d):
                # compute elementary matrix
                X_elem = self._s[i] * np.outer(self._U[:,i], self._VT[i,:])
                # flip rows
                X_rev = X_elem[::-1]
                # compute diagonal average and store them as columns
                self._ts_components[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
    
    def _reconstruct(self):
        """
        Computes the elementary components of the data from the SVD. Algorithm 2
        of Golyandina et al. (2014) is applied.
        """
        print "Computing reconstruction..."
        assert hasattr(self, "_k")
        assert hasattr(self, "_l")
        assert hasattr(self, "_d")
        assert hasattr(self, "_s")
        assert hasattr(self, "_U")
        assert hasattr(self, "_VT")
        
        if self._ndim == 1:
            # (step 1 omitted)
            # precompute weights (step 2)
            if not hasattr(self, "_weights"):
                self._compute_weights()

            # allocate array
            self._ts_components = np.zeros((self._n, self._d))
            Uzeros = np.zeros(self._k - 1)
            Vzeros = np.zeros(self._l - 1)
            # loop over eigentriples
            from scipy.fftpack import fft, ifft
            for i in range(self._d):
                # extend eigenvectors (step 3)
                U = np.hstack((self._U[:,i], Uzeros))
                assert U.size == self._n
                V = np.hstack((self._VT[i,:], Vzeros))
                assert V.size == self._n
                # compute component (steps 4,5 and 6)
                self._ts_components[:,i] = self._s[i] * ifft(fft(U) * fft(V)).real / self._weights
        else:
            L = np.prod(self._l)
            K = np.prod(self._k)
            
            # precompute weights (step 1)
            weights = diagsums(np.ones(L), np.ones(K), self._n, self._l)
            
            # compute shape (step 2) is omitted because it is only for shaped SSA
            
            # allocate array
            self._ts_components = np.zeros(self._n + (self._d, ))
            # loop over eigentriples
            for i in range(self._d):
                # compute component (steps 3 and 4)
                self._ts_components[:,:,i] = self._s[i] * \
                    diagsums(self._U[:,i], self._VT[i,:], self._n, self._l) / weights

    def _compute_group_spectra(self):
        """
        Computes spectrum of the groups of the signal.
        """
        print "Computing group spectra..."
        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        assert hasattr(self, "_orig_ts")
        
        if self._ndim > 1:
            raise NotImplementedError()
        
        # determine sampling time
        dt = np.unique(np.diff(self._orig_ts.index.values))
        assert np.allclose(dt, dt[0])
        dt = dt[0]
        print "   sampling time: {0:.3f}".format(dt)
        
        # compute sampling frequency
        fs = 1.0 / dt
        print "   sampling frequency: {0:.3f}".format(fs)
        
        # compute spectrum
        self._group_spectra, self._freq = compute_spectrum(self._ts_groups, fs)
        assert self._group_spectra.ndim == 2
        assert self._group_spectra.shape[1] == self._ts_groups.shape[1]
        assert self._group_spectra.shape[0] == self._freq.shape[0]
        print "   Shannon frequency: {0:.3f}".format(self._freq.max())
    
    def _compute_correlation(self):
        """
        Computes the correlation matrix for the decomposed time series.
        """
        print "Computing correlations..."
        assert hasattr(self, "_k")
        assert hasattr(self, "_l")
        assert hasattr(self, "_ts_components")
        
        # calculate the weights
        if not hasattr(self, "_weights"):
            self._compute_weights()
        
        # inline function for weighted inner product
        if self._ndim == 1:
            w_inner = lambda i, j: self._weights.dot(self._ts_components[...,i]*self._ts_components[...,j])
        elif self._ndim == 2:
            w_inner = lambda i, j: np.dot(self._weights[0],
                                          (self._ts_components[...,i] * self._ts_components[...,j]).dot(self._weights[1]))
        
        # calculated inverted weighted norms, 1 / ||F_i||_w
        inv_wnorms = np.array([w_inner(i, i) for i in range(self._d)])
        inv_wnorms = inv_wnorms**-0.5
        
        # calculate correlation matrix
        self._wcorr = np.identity(self._d)
        for i in range(self._d):
            # make use of symmetry
            for j in range(i + 1, self._d):
                self._wcorr[i,j] = abs(w_inner(i,j)) * inv_wnorms[i] * inv_wnorms[j]
                self._wcorr[j,i] = self._wcorr[i,j]

    def _compute_powers(self):
        """
        Computers relative and cummulative powers of the components.
        
        Returns
        ----------
        rel_power : numpy.ndarray
            Relative power of the components.
        cum_power : numpy.ndarray
            Cummulative powers of the components.
        cum_power : float
            Total power.
        """
        assert hasattr(self, "_s")
        
        total_power = (self._s**2).sum()
        rel_power = self._s**2 / total_power
        cum_power = (self._s**2).cumsum() / total_power 
        
        return rel_power, cum_power, total_power
   
    def _compute_svd(self, method="svd", min_components=16):
        """
        Computes the singular value decomposition (SVD) of the trajectory matrix.
        
        Parameters
        ----------
        method : str
            The method chosen to compute or approximate the SVD.
        n_components : int
            Number of singular values to compute.
        """
        assert hasattr(self, "_X")
        assert isinstance(method, str)
        assert method in ("svd", "svds", "rand_svd")
        
        if method is "svd":
            from scipy.linalg import svd
            print "Computing SVD using direct method..."
            self._U, self._s, self._VT = svd(self._X)
            self._d = np.count_nonzero(np.abs(self._s) > self.__tol)

        elif method is "svds":
            from scipy.sparse.linalg import svds
            print "Computing SVD using iterative method..."
            assert isinstance(min_components, int) and min_components >= 6
            k = min_components
            ratio = 1.0
            i = 0
            while ratio > self.__relative_ratio and k < self._l // 2:
                print "   On iteration {0} (k = {1})...".format(i, k)
                self._U, self._s, self._VT = svds(self._X, k=k, tol=self.__tol)
                k *= 2
                ratio = self._s.min() / self._s.max()
                i += 1
            sort_ind = np.argsort(self._s)[::-1]
            self._s = self._s[sort_ind]
            self._U = self._U[:,sort_ind]
            self._VT = self._VT[sort_ind,:]
            self._d = self._s.size
        
        elif method is "rand_svd":
            from sklearn.utils.extmath import randomized_svd
            print "Computing SVD using randomized method..."
            self._U, self._s, self._VT= randomized_svd(self._X, min_components,
                                                       n_oversamples=15)
            self._d = min_components

    def _compute_weights(self):
        """
        Computes the inner product weights by summing the diagonals of a matrix, which contains only ones.
        """
        assert hasattr(self, "_l")
        assert hasattr(self, "_n")
        assert hasattr(self, "_ndim")
        
        if self._ndim == 1:
            self._weights = np.array([
                min(i + 1, self._l, self._k, self._n - i)
                for i in range(self._n)])
        else:
            assert len(self._k) == self._ndim
            assert len(self._l) == self._ndim
            assert len(self._n) == self._ndim
            
            self._weights = []
            
            for j in range(self._ndim):
                self._weights.append(np.array([
                    min(i + 1, self._l[j], self._k[j], self._n[j] - i) 
                    for i in range(self._n[j]) ]))

    def _embed(self):
        """
        Creates the trajectory matrix from input data.
        """
        print "Computing trajectory matrix..."
        assert hasattr(self, "_orig_ts")
        assert hasattr(self._orig_ts, "values")
        ndim = self._orig_ts.values.ndim
        values = self._orig_ts.values
        
        if ndim == 1:
            assert hasattr(self, "_k")
            assert hasattr(self, "_l")
            self._X = elementary_embedding(values, self._l)
        elif ndim == 2:
            assert hasattr(self, "_k")
            assert hasattr(self, "_l")
            assert isinstance(self._k, (tuple, list))
            assert isinstance(self._l, (tuple, list))
            assert len(self._k) == 2 and len(self._l) == 2
            
            tmpX = [elementary_embedding(values[:,i], self._l[0]) for i in range(0, self._n[1])]
            self._X = block_hankel((self._l[1], self._k[1]), tmpX)
            assert self._X.shape == (np.prod(self._l), np.prod(self._k))

        else:
            raise NotImplementedError()
    
    def _group(self, cutoff_index=64, algorithm="DBSCAN"):
        """
        Groups elementary components using clustering algorithms.
        
        Parameters
        ----------
        algorithm : str
            String specifying the clustering algorithm to be applied. Possible
            is ``DBSCAN`` or ``AffinityPropagation``.
        """
        print "Computing clustering with cutoff at i = {}...".format(cutoff_index)
        assert hasattr(self, "_wcorr")
        assert isinstance(algorithm, str) 
        assert algorithm in ("DBSCAN", "AffinityPropagation")
        
        # compute distance from correlation
        X = np.abs(self._wcorr[:cutoff_index,:cutoff_index] - 1.0)
        
        if algorithm is "DBSCAN":
            from sklearn.cluster import DBSCAN
            db = DBSCAN(min_samples=2, metric="precomputed")
            db.fit(X)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)                    
        
        elif algorithm is "AffinityPropagation":
            from sklearn.cluster import AffinityPropagation
            af = AffinityPropagation(affinity="precomputed")
            labels = af.labels_
            cluster_centers_indices = af.cluster_centers_indices_
            n_clusters = len(cluster_centers_indices)

        # print user info
        print "   Estimated number of clusters: {}".format(n_clusters)
        if algorithm is "DBSCAN":
            print "   Estimated number of noise points: {}".format(n_noise)
        from sklearn import metrics
        s_score = metrics.silhouette_score(X, labels, metric="precomputed")
        print "   Silhouette Coefficient: {:0.3f}".format(s_score)
        
        # extract cluster indices and power
        clusters = dict()
        _, _, total_power = self._compute_powers()
        for i in range(n_clusters):
            ind = np.where(labels == i)[0]
            relative_power = (self._s[ind]**2).sum() / total_power
            if relative_power > self.__power_threshold:
                print "   Relative power of cluster {0}: {1:0.3f}".format(i, relative_power)
            clusters[i] = [tuple(ind), relative_power]
        
        # extract clusters from noise
        noise_ind = np.where(labels == -1)[0]
        if algorithm is "DBSCAN":
            assert noise_ind.size == n_noise
        j = n_clusters
        for i, ind in enumerate(noise_ind):
            relative_power = self._s[ind]**2 / total_power
            if relative_power > self.__power_threshold:
                print "   Relative power of noise {0}: {1:0.3f}".format(i, relative_power)
                clusters[j] = [ind, relative_power]
                j += 1

        # set final number of clusters
        self._n_groups = len(clusters)
        
        # sort according to power
        powers = np.array(zip(*clusters.values())[1])
        sort_ind = np.argsort(powers)[::-1]
        self._group_power = powers[sort_ind]
        
        # compute groups
        if self._ndim == 1:
            self._ts_groups = np.zeros((self._n, self._n_groups))
        else:
            self._ts_groups = np.zeros(self._n + (self._n_groups, ))
        for i, ind in enumerate(sort_ind):
            indices = clusters[ind][0]
            if isinstance(indices, (tuple, list, np.ndarray)):
                if len(indices) > 1:
                    self._ts_groups[...,i] = self._ts_components[...,indices].sum(axis=-1)
                else:
                    self._ts_groups[...,i] = self._ts_components[...,indices]
            else:
                self._ts_groups[...,i] = self._ts_components[...,indices]

    def get_components(self, n_components=None, pandas=False):
        """
        Returns all the time series components in a single object.
        
        Parameters
        ----------
        n_components : int
            Number of components to include. Default value is the maximum number of components.
        
        pandas : pandas
            Boolean for returning pandas.DataFrame object.
            
        Returns
        ----------
        components : np.ndarray, pandas.DataFrame
            Components as an array or data frame object.
        """
        assert isinstance(n_components, int) and n_components>=0
        assert hasattr(self, "_ts_components")
        assert hasattr(self, "_d")
        
        if n_components is None:
            n_components = self._d
        else:
            n_components = min(n_components, self._d)
            
        # create list of columns
        cols = [i for i in range(n_components)]
        
        if pandas and self._ndim == 1:
            return pd.DataFrame(self._ts_components[...,:n_components],
                                columns=cols, index=self._orig_ts.index)
        elif pandas:
            return pd.DataFrame(self._ts_components[...,:n_components])
        else:
            return self._ts_components[...,:n_components]
    
    def get_groups(self, max_index=None, min_index=0, pandas=False):
        """
        Returns all the time series groups in a single Pandas DataFrame object.
        
        Parameters
        ----------
        max_index : int
            Maximum index of the group to be included. Default value is the number of groups.
        
        min_index : int
            Minimum index of the group to be included. Default value is zero.
        
        pandas : bool
            Boolean for returning pandas.DataFrame object.
            
        Returns
        ----------
        group_spectra : np.ndarray, pandas.DataFrame
            Groups as an array or data frame object.
        """
        assert isinstance(min_index, int) and min_index >= 0
        assert isinstance(pandas, bool)
        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        
        if max_index is None:
            max_index = self._n_groups
        else:
            assert isinstance(max_index, int) and max_index >= min_index
            max_index = min(max_index, self._n_groups)
            
        # create list of columns - call them F0, F1, F2, ...
        cols = [i for i in range(min_index, max_index)]
        
        if pandas and self._ndim == 1:
            return pd.DataFrame(self._ts_groups[...,min_index:max_index], columns=cols,
                                index=self._orig_ts.index)
        elif pandas:
            return pd.DataFrame(self._ts_groups[...,min_index:max_index])
        else:
            return self._ts_groups[...,min_index:max_index]
    
    def get_group_spectra(self, n_groups=None, pandas=False):
        """
        Returns the spectra of the groups in a single Pandas DataFrame object.
        
        Parameters
        ----------
        n_groups : int
            Number of groups to include. Default value is the maximum number of groups.
            
        pandas : bool
            Boolean for returning pandas.DataFrame object.

        Returns
        ----------
        group_spectra : np.ndarray, pandas.DataFrame
            Group spectra as an array or data frame object.
        """
        assert isinstance(n_groups, int) and n_groups >= 0
        assert isinstance(pandas, bool)
        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        
        if n_groups is None:
            n_groups = self._n_groups
        else:
            n_groups = min(n_groups, self._n_groups)
            
        if not hasattr(self, "_group_spectra"):
            self._compute_group_spectra()
        
        # create list of columns
        cols = [i for i in range(n_groups)]
        
        if pandas and self._ndim == 1:
            return pd.DataFrame(self._group_spectra[...,:n_groups],
                                columns=cols, index=self._freq)
        elif pandas:
            return pd.DataFrame(self._group_spectra[...,:n_groups])
        else:
            return self._group_spectra[...,:n_groups]

    def reconstruct_from_groups(self, indices, pandas=False):
        """
        Reconstructs the time series from its groups, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices : int, tuple, list or slice
            Object representing the groups to sum.
            
        pandas : bool
            Boolean for returning pandas.DataFrame object.
            
        Returns
        ----------
        reconstructed_data : np.ndarray, pandas.DataFrame
            Reconstructed data as an array or data frame object.
        """
        assert isinstance(indices, (int, list, tuple, slice))
        assert isinstance(pandas, bool)
        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        
        if isinstance(indices, int):
            assert indices < self._n_groups
            indices = [indices]

        elif isinstance(indices, (list, tuple)):
            assert min(indices) <= max(indices)
            assert min(indices) >= 0 and max(indices) < self._n_groups

        ts_vals = self._ts_groups[...,indices].sum(axis=-1)
        
        if pandas and self._ndim == 1:
            return pd.DataFrame(ts_vals, index=self._orig_ts.index)
        elif pandas:
            return pd.DataFrame(ts_vals)
        else:
            return ts_vals
    
    def reconstruct_elementary(self, indices, pandas=False):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices : int, tuple, list or slice
            Object representing the elementary components to sum.
            
        pandas : bool
            Boolean for returning pandas.DataFrame object.

        Returns
        ----------
        reconstructed_data : np.ndarray, pandas.DataFrame
            Reconstructed data as an array or data frame object.
        """
        assert isinstance(indices, (int, list, tuple, slice))
        assert isinstance(pandas, bool)
        assert hasattr(self, "_ts_components")

        if isinstance(indices, int):
            assert indices < self._d
            indices = [indices]

        elif isinstance(indices, (list, tuple)):
            assert max(indices) < self._d

        ts_vals = self._ts_components[...,indices].sum(axis=-1)
        
        if pandas and self._ndim == 1:
            return pd.Series(ts_vals, index=self._orig_ts.index)
        elif pandas:
            return pd.Series(ts_vals)
        else:
            return ts_vals
    
    def plot_elementary_matrices(self, max_index=12, min_index=0):
        """
        Plots the correlation matrix for the decomposed time series.
        
        Parameters
        ----------
        max_index : int
            Maximum index of the elementary matrix shown in the plot.
        min_index : int
            Minimum index of the elementary matrix shown in the plot.
        """
        assert isinstance(min_index, int) and min_index >= 0
        assert isinstance(max_index, int) and min_index < max_index

        assert hasattr(self, "_d")

        if max_index < 4:
            max_index = 4
        else:
            max_index = min(self._d, max_index)
        assert max_index < self._d
        
        if not hasattr(self, "_X_elem"):
            self._compute_components(save_mem=False, max_index=max_index)
            
        ncols = 4
        nrows = (max_index - min_index) // ncols
        if (max_index - min_index) % ncols > 0:
            nrows += 1

        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(min_index, max_index):
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(hankelize(self._X_elem[i]))
            plt.xticks([])
            plt.yticks([])
            plt.title(r"$\tilde{\mathbf{X}}_{" + str(i) + "}$")
        plt.tight_layout()
    
    def plot_groups(self, max_index=6, min_index=0, indices=None):
        """
        Plots the grouped components.
        
        Parameters
        ----------
        max_index : int
            Maximum group index to plot.
        min_index : int
            Minimum group index to plot.
        indices : tuple, list
            List of indices.
        """
        # TODO: Does not work for a field!
        assert isinstance(min_index, int) and min_index >= 0
        assert isinstance(max_index, int) and min_index < max_index
        
        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        
        max_index = min(max_index, self._n_groups)
        
        if indices is not None:
            assert isinstance(indices, (list, tuple))
            assert all(isinstance(i, int) for i in indices)
            assert min(indices) <= max(indices)
            assert min(indices) >= 0 and max(indices) <= self._n_groups
            groups = self.get_groups(max_index=max(indices), min_index=min(indices)).loc[:,indices]
            
        else:
            groups = self.get_groups(max_index=max_index, min_index=min_index)
        
        rename_columns(groups, group_labels=True)
        
        # plot groups
        allaxes = groups.plot(subplots=True, layout=(groups.shape[1], 1))
        for ax in allaxes.flatten():
            ax.grid(which="both")
        
        # plot original time series
        if min_index == 0:
            self._orig_ts.plot(ax=allaxes.flatten()[0], alpha=0.75)
        
    def plot_group_reconstruction(self, indices=None):
        """
        Plots the signal reconstructed by the group indices.
        
        Parameters
        ----------
        indices : int, tuple, list or slice
            Object representing the groups to sum.
        """
        # TODO: Does not work for a field!
        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        
        if indices is None:
            indices = range(self._n_groups)
        
        data_frame = pd.DataFrame({"original": self._orig_ts, 
                                   "reconstructed": self.reconstruct_from_groups(indices)})
        # plot groups
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        alpha=[0.5, 1.0]
        linewidths = [1.0, 2.0]
        for col, a, lw in zip(data_frame.columns, alpha, linewidths):
            data_frame[col].plot(alpha=a, lw=lw, ax=ax, label=col)
            
    def plot_group_frequencies(self):
        """
        Plots the most dominant period of all groups.
        """
        # TODO: Does not work for a field!
        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        
        spectra = self.get_group_spectra(self._n_groups)
        
        freq = []
        for s in spectra:
            ind = np.argmax(spectra[s].values)
            freq.append(spectra[s].index[ind])
        periods = 1. / np.array(freq)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(periods, 'x')
        ax.plot(np.ones(periods.shape), '--')
        ax.set_xticks(np.arange(0, periods.size, 2))
        ax.set_yticks(np.arange(0, periods.max(), 2))
        ax.grid()
        
        return fig, ax
    
    def plot_group_spectra(self, max_index=6, xlim=(0., 2.), indices=None):
        """
        Plots the Fourier spectrum of the grouped components.
        
        Parameters
        ----------
        max_index : int
            Maximum group index to plot.
        """
        # TODO: Does not work for a field!
        assert isinstance(max_index, int) and max_index>=0
        assert isinstance(xlim, (tuple, list))
        assert len(xlim) == 2
        assert xlim[0] < xlim[1]

        assert hasattr(self, "_ts_groups")
        assert hasattr(self, "_n_groups")
        max_index = min(max_index, self._n_groups)
        
        if indices is not None:
            assert isinstance(indices, (list, tuple))
            assert all(isinstance(i, int) for i in indices)
            assert min(indices) <= max(indices)
            assert min(indices) >= 0 and max(indices) <= self._n_groups
            spectra = self.get_group_spectra(max(indices) + 1).loc[:,indices]
            
        else:
            spectra = self.get_group_spectra(max_index)

        rename_columns(spectra, group_labels=True)
        
        # plot all spectra
        allaxes = spectra.plot(subplots=True, layout=(spectra.shape[1], 1),
                               xlim=xlim)
        for ax in allaxes.flatten():
            ax.grid(which="major")
        
        # add secondary axis for period of the signal
        ax = allaxes.flatten()[0]
        axx = ax.twiny()
        axx.set_xlim(ax.get_xlim())
        axx.set_xticks(ax.get_xticks()[1:])
        axx.set_xticklabels(1./ax.get_xticks()[1:])
    
    def plot_correlation(self, min_index=0, max_index=None):
        """
        Plots the correlation matrix for the decomposed time series.
        
        Parameters
        ----------
        min_index : int
            Minimum index of the correlation matrix shown in the plot.
        max_index : int
            Maximum index of the correlation matrix shown in the plot.
        """
        assert isinstance(min_index, int) and min_index >= 0
        assert hasattr(self, "_d")
        
        if min_index is None:
            min_index = 0
        
        if max_index is None:
            max_index = self._d
        else:
            assert isinstance(max_index, int) and max_index > min_index
            max_index = min(max_index, self._d)
        
        if not hasattr(self, "_wcorr"):
            self._compute_correlation()
        
        # create plot
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.imshow(self._wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$w_{i,j}$")
        plt.clim(0,1)
        
        # adjust axes limits for plotting purposes
        if max_index == self._d:
            max_rnge = self._d - 1
        else:
            max_rnge = max_index
        plt.xlim(min_index - 0.5, max_rnge + 0.5)
        plt.ylim(max_rnge + 0.5, min_index - 0.5)
        
    def plot_powers(self, max_index=12):
        """
        Plots the contributions to the signal.
        
        Parameters
        ----------
        max_index : int
            Maximum index in the plot.
        """
        assert isinstance(max_index, int) and 0 < max_index
        assert hasattr(self, "_d")
        assert hasattr(self, "_s")
        
        max_index = min(self._d, max_index)
        
        relative_power, cummulative_power, _ = self._compute_powers()

        # create plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14,5))
        ax[0].plot(relative_power)
        ax[0].set_xlim(0, max_index)
        ax[0].set_title("relative contribution of $\mathbf{X}_i$ to trajectory matrix")
        ax[0].set_xlabel("$i$")
        ax[0].set_ylabel("contribution (%)")
        ax[1].plot(cummulative_power)
        ax[1].set_xlim(0, max_index)
        ax[1].set_title("cumulative contribution of $\mathbf{X}_i$ to trajectory matrix")
        ax[1].set_xlabel("$i$")
        ax[1].set_ylabel("contribution")
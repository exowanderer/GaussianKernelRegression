import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy.spatial import cKDTree
from tqdm import tqdm

def find_qhull_one_point(point, x0, y0, np0, inds):
    ''' Compute the exponentially weighted distance to each point, as compared to its neighbors
        
        This produces the weighting function for significance in the Gaussian kernel regression.
        Allowing the GKR to scale to very large data sets by prioritizing the data points 'near' the source point
        
        Paramaeters
        -----------
        point (int): index of source point within: x0, y0, np0, inds
        x0 (1dArray): x-positions of source over time
        y0 (1dArray): y-positions of source over time
        np0 (1dArray): PSF width of source over time ('beta pixels')
        inds (list): list of data points `near` to source point, nominally determined by kdtree
        
        Returns
        -------
        gw_temp (float) Gaussian weighted summation of qhull volume (i.e. 'weight') of source per frame
    '''
    
    dx = x0[inds[point]] - x0[point]
    dy = y0[inds[point]] - y0[point]
    
    if np0.sum() != 0.0:
        dnp = np0[inds[point]] - np0[point]
    
    sigx = np.std(dx)
    sigy = np.std(dy)
    
    if dnp.sum() != 0.0:
        signp = np.std(dnp)
        exponent = -dx**2./(2.0*sigx**2.) + -dy**2./(2.*sigy**2.) + -dnp**2./(2.*signp**2.)
    else:
        exponent = -dx**2./(2.0*sigx**2.) + -dy**2./(2.*sigy**2.)
    
    gw_temp = np.exp(exponent)
    
    return gw_temp / gw_temp.sum()

def gaussian_weights(xpos, ypos, npix = None, inds = None, n_nbr = 50, return_inds=False,
                      a = 1.0, b = 0.7, c = 1.0, expansion = 1000., ncores=1):
    '''
        Python Implimentation of N. Lewis method, described in Lewis etal 2012, Knutson etal 2012, Fraine etal 2013
        
        Taken from N. Lewis IDL code:
            
            Construct a 3D surface (or 2D if only using x and y) from the data
            using the qhull.pro routine.  Save the connectivity information for
            each data point that was used to construct the Delaunay triangles (DT)
            that form the grid.  The connectivity information allows us to only
            deal with a sub set of data points in determining nearest neighbors
            that are either directly connected to the point of interest or
            connected through a neighboring point
        
        Python Version:
            J. Fraine    first edition, direct translation from IDL 12.05.12
            
        Paramaeters
        -----------
        xpos (1Darray): x-positions of source over time
        ypos (1Darray): y-positions of source over time
        npix (1dArray): PSF width of source over time (Defaut: None)
        inds (list): List of lists of indices 'near' each source over time (Defaut: None)
        n_nbr (int): Number of points to be considered by cKDTree (Defaut: 50)
        return_inds (bool): Toggle whether to return the cKDTree indices; could speed up future use (Defaut: False)
        a (float): Exponential scale under x-distances in qhull (Defaut: 1.0)
        b (float): Exponential scale under y-distances in qhull (Defaut: 0.7)
        c (float): Exponential scale under 'width-distances' in qhull (Defaut: 1.0)
        expansion (float): Scale factor to avoid loss of significance (Defaut: 1000)
        ncores (int): Number of cores to use with `multiprocessing.Pool` (Defaut: 1)
        
        Returns
        -------
        gw_list (1Darry): Gaussian weighted summations of qhull volume (i.e. 'weights') over source over time
        inds (list): List of lists for indices `near` each point (from `cKDTree`) over time
    '''
    
    # The surface fitting performs better if the data is scattered about zero
    x0  = (xpos - np.median(xpos))/a
    y0  = (ypos - np.median(ypos))/b
    
    if npix is not None and bool(c):
        np0 = np.sqrt(npix)
        np0 = (np0 - np.median(np0))/c
        features  = np.transpose((y0, x0, np0))
    else:
        features  = np.transpose((y0, x0))
        
        if np.sum(np0) == 0.0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because Noise Pixels are Zero')
        if c == 0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because c == 0')
    
    if inds is None:
        kdtree = cKDTree(features * expansion)  # Multiplying `features` by 1000.0 avoids precision problems
        inds = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:]
        
        print('WARNING: Because `inds` was not provided, we must now compute and return it here')
        return_inds= True
    
    n, k = inds.shape  # This is the number of nearest neighbors you want
    
    func = partial(find_qhull_one_point, x0=x0, y0=y0, np0=np0, inds=inds)
    
    if ncores > 1:
        print('Computing weights with Pool; no status update will be presented')
        pool = Pool(ncores)
        
        gw_list = pool.starmap(func, zip(range(n)))
        
        pool.close()
        pool.join()
    else:
        gw_list = []
        for idx in tqdm(range(n), total=n):
            gw_list.append(func(idx))
    
    if return_inds:
        return np.array(gw_list), inds
    else:
        return np.array(gw_list)

def gaussian_kernel_regression(residuals, gaussian_weights, indices):
    ''' Compute the GKR `residuals` values over time
        
        This produces the comparison to the actual residuals data, to be used iterative with a least-sq or MCMC solver
        
        Paramaeters
        -----------
        residuals (1Darray): residual source value over time (to be GKR regressed)
        gaussian_weights (1Darray): set of Gaussian weights per source point to compare with neighbor
        indices (1dArray): KDTree list of lists for nearby indicies per source
        
        Returns
        -------
        prediction (1Darray) Gaussian kernel regression prediction for the value of the source over time
    '''
    return np.sum(residuals[indices] * gaussian_weights, axis=1)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    nPts = int(1e5)
    
    xpos = 0.35*np.sin(np.arange(0, nPts) / 1500 + 0.5) + 15 + np.random.normal(0, 0.2, nPts)
    ypos = 0.35*np.sin(np.arange(0, nPts) / 2000 + 0.7) + 15 + np.random.normal(0, 0.2, nPts)
    npix = 0.25*np.sin(np.arange(0, nPts) / 2500 + 0.4) + 15 + np.random.normal(0, 0.2, nPts)
    residuals = 1 + 0.01*(xpos - xpos.mean()) + 0.01*(ypos - ypos.mean()) + 0.01*(npix - npix.mean())
    
    n_nbr  = 50
    points = np.transpose([xpos, ypos, npix])
    kdtree = cKDTree(points)
    
    ind_kdtree = kdtree.query(kdtree.data, n_nbr + 1)[1][:, 1:] # skip the first one because it's the current point
    
    # `gaussian_weights` only returns the gaussian weights in the indices are provided
    gw_kdtree   = gkr.gaussian_weights(xpos, ypos, npix, ind_kdtree)
    
    # gaussian_kernel_regression computes the prediction for the GKR over the residuals
    gkr_kdtree = gkr.gaussian_kernel_regression(residuals, gw_kdtree, ind_kdtree)
    
    fig1, ax1 = plt.subplots(1,1)
    ax1.plot(residuals , '.', ms=1, alpha=0.5)
    ax1.plot(gkr_kdtree , '.', ms=1, alpha=0.5)
    
    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(residuals - gkr_kdtree , '.', ms=1, alpha=0.5)
    ax2.set_title('Scipy.cKDTree Gaussian Kernel Regression')
    ax2.set_ylim(-0.0005,0.0005)

# from scipy.spatial      import cKDTree
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def decorrelate_gk_transit(times, phots, params, gk, nbr_ind):
    '''
        form gaussian weighting fuction to decorelate photometry using a gaussian kernel `gk`
    '''
    nbr_ind = np.array(nbr_ind, int).copy()
    tmodel  = transit_model_wrap(times, params)
    return np.sum((phots / tmodel)[nbr_ind]*gk,axis=1)

def decorrelate_gk_eclipse(times, phots, params, gk, nbr_ind):
    '''
        form gaussian weighting fuction to decorelate photometry using a gaussian kernel `gk`
    '''
    raise Exception("\n\n *** NEED TO UPDATE `decorrelate_gk_eclipse` *** \n\n")
    nbr_ind = np.array(nbr_ind, int).copy()
    # emodel  = NETransit.eclipse_model(times, params)
    return np.sum((phots / emodel)[nbr_ind]*gk,axis=1)

def decorrelate_gk_functionless(phots, gk, nbr_ind):
    '''
        form gaussian weighting fuction to decorelate photometry using a gaussian kernel `gk`
    '''
    nbr_ind = np.array(nbr_ind, int).copy()
    return np.sum(phots[nbr_ind]*gk,axis=1)

def gk_weighting_wrapper(gk_params):
    times, flux, gk, nbr = gk_params
    return lambda transit_params: decorrelate_gk(times, flux, transit_params, gk, nbr)

def find_nbr_qhull(xpos, ypos, npix, sm_num = 100, a = 1.0, b = 1.0, c = 1.0, print_space = 10000.):
    '''
        Python Implimentation of N. Lewis method, described in Lewis etal 2012, Knutson etal 2012

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
    '''
    neigh = NearestNeighbors(n_neighbors=sm_num) # need 1 more to exclude the test point itself
    #The surface fitting performs better if the data is scattered about zero
    
    npix    = np.sqrt(npix)
    
    x0  = (xpos - np.median(xpos))/a
    y0  = (ypos - np.median(ypos))/b
    
    if np.sum(npix) != 0.0 and c != 0:
        np0 = (npix - np.median(npix))/c
    else:
        if np.sum(npix) == 0.0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because Noise Pixels are Zero')
        if c == 0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because c == 0')
    
    k            = sm_num                           # This is the number of nearest neighbors you want
    n            = x0.size                          # This is the number of data points you have
    nearest      = np.zeros((k,n),dtype=np.int64)   # This stores the nearest neighbors for each data point
    
    #Multiplying by 1000.0 avoids precision problems
    if npix.sum() != 0.0 and c != 0:
        # kdtree  = cKDTree(np.transpose((y0*1000., x0*1000., np0*1000.)))
        neigh.fit(np.transpose((y0*1000., x0*1000., np0*1000.)))
    else:
        # kdtree  = cKDTree(np.transpose((y0*1000., x0*1000.)))
        neigh.fit(np.transpose((y0*1000., x0*1000.)))
    
    inds= neigh.kneighbors(return_distance=False)
    
    gw  = np.zeros((k,n),dtype=np.float64) # This is the gaussian weight for each data point determined from the nearest neighbors
    
    start   = time.time()
    for point in tqdm(range(n), total=n):
        # if np.round(point/print_space) == point/print_space: print_timer(point, n, start)
        
        # ind         = kdtree.query(kdtree.data[point],sm_num+1)[1][1:]
        ind         = inds[point]#neigh.kneighbors(points[point:point+1])[1].flatten()[1:]
        
        dx          = x0[ind] - x0[point]
        dy          = y0[ind] - y0[point]
        
        if npix.sum() != 0.0 and c != 0:
            dnp         = np0[ind] - np0[point]
        
        sigx        = np.std(dx )
        sigy        = np.std(dy )
        if npix.sum() != 0.0 and c != 0:
            signp       = np.std(dnp)
        if npix.sum() != 0.0 and c != 0:
            gw_temp     = np.exp(-dx**2./(2.0*sigx**2.)) * \
                          np.exp(-dy**2./(2.*sigy**2.))  * \
                          np.exp(-dnp**2./(2.*signp**2.))
        else:
            gw_temp     = np.exp(-dx**2./(2.0*sigx**2.)) * \
                          np.exp(-dy**2./(2.*sigy**2.))
        
        gw_sum      = gw_temp.sum()
        gw[:,point] = gw_temp/gw_sum
        
        if (gw_sum == 0.0) or ~np.isfinite(gw_sum):
            raise Exception('(gw_sum == 0.0) or ~isfinite(gw_temp))')
        
        nearest[:,point]  = ind
    
    return gw.transpose(), nearest.transpose() # nearest  == nbr_ind.transpose()

def GaussianKernel_KDtree(statevectors, knobs = None, sm_num = 100, expansion = 1e3, print_space = 1):
    '''
        :: INPUTS ::
            statevectors : an nDarray of `n` 1Darrays to use for the gaussian kernel
            knobs        : 1Darray of floats corresponding to the weighting factors for each state vector
            sm_num       : number of nearest neighbors to be computed
            expansion    : factor by which to avoid loss of significance
            print_space  : number of iterations to print after

        :: OUTPUTS ::
            gaussian_kernel : gaussian kernel function to use with gaussian weighting decorrelation routine
            nearest         : nearest neighbors arrays to use with gaussian weighting decorrelation routine
    '''
    '''
        Python Implimentation of N. Lewis method, described in Lewis etal 2012, Knutson etal 2012

        Converted from N. Lewis IDL code:

            Construct a 3D surface (or 2D if only using x and y) from the data
            using the qhull.pro routine.  Save the connectivity information for
            each data point that was used to construct the Delaunay triangles (DT)
            that form the grid.  The connectivity information allows us to only
            deal with a sub set of data points in determining nearest neighbors
            that are either directly connected to the point of interest or
            connected through a neighboring point

        Python Version:

        J. Fraine    first edition -- direct translation from IDL 12.05.12

        The surface fitting performs better if the data is scattered about zero
    '''
    
    nStates = statevectors.shape[0] # This is the number of state vectors you have
    nPts    = statevectors.shape[1] # This is the number of data points you have
    
    if knobs == None:
        knobs = np.ones(nStates)
    
    NormedStateVectors = np.zeros(statevectors.shape)
    
    #expansion = 1e3
    for s in range(nStates):
        NormedStateVectors[s] = (statevectors[s] - np.median(statevectors[s])) / knobs[s]
    
    nearest = np.zeros((sm_num,nPts),dtype=np.int64)   # This stores the nearest neighbors for each data point
    # kdtree  = cKDTree(NormedStateVectors.T*expansion)
    neigh = NearestNeighbors(n_neighbors=sm_num) # need 1 more to exclude the test point itself
    neigh.fit(NormedStateVectors.T*expansion)
    
    gaussian_kernel  = np.zeros((sm_num,nPts),dtype=np.float64) # This is the gaussian weight for each data point determined from the nearest neighbors
    
    start   = time.time()
    for point in tqdm(range(nPts), total=nPts):
        # nearest[:,point] = kdtree.query(kdtree.data[point],sm_num+1)[1][1:]
        nearest     = neigh.kneighbors(return_distance=False)#NormedStateVectors[k])[1].flatten()[1:]
        dstates     = np.zeros((nStates, sm_num))
        sigstates   = np.zeros((nStates, sm_num))
        for state in range(nStates):
            dstates[state]      = statevectors[state][nearest[:,point]] - statevectors[state][point]
            sigstates[state]    = np.std(dstates[state])
        
        gk_temp = np.zeros(sm_num)
        for state in range(nStates):
            gk_temp += -0.5*(dstates[state]/sigstates[state])**2.0
        
        gk_temp     = np.exp(gk_temp)
        gk_sum      = gk_temp.sum()
        gaussian_kernel[:,point] = gk_temp/gk_sum
        
        if (gk_sum == 0.0) or ~np.isfinite(gk_sum):
            raise Exception('(gw_sum == 0.0) or ~isfinite(gw_temp))')
    
    return gaussian_kernel.transpose(), nearest.transpose() # nearest  == nbr_ind.transpose()

from scipy.spatial      import cKDTree

def find_qhull_one_point(point, x0, y0, np0, inds, sm_num):
    ind         = inds[point]#kdtree.query(kdtree.data[point],sm_num+1)[1][1:]
    dx          = x0[ind] - x0[point]
    dy          = y0[ind] - y0[point]
    
    if np0.sum() != 0.0:# and c != 0:
        dnp         = np0[ind] - np0[point]
    
    sigx        = np.std(dx )
    sigy        = np.std(dy )
    if dnp.sum() != 0.0:# and c != 0:
        signp       = np.std(dnp)
    if dnp.sum() != 0.0:# and c != 0:
        gw_temp     = np.exp(-dx**2./(2.0*sigx**2.)) * \
                      np.exp(-dy**2./(2.*sigy**2.))  * \
                      np.exp(-dnp**2./(2.*signp**2.))
    else:
        gw_temp     = np.exp(-dx**2./(2.0*sigx**2.)) * \
                      np.exp(-dy**2./(2.*sigy**2.))
    
    gw_sum      = gw_temp.sum()
    #gw[:,point] = gw_temp/gw_sum
    
    # if (gw_sum == 0.0) or ~np.isfinite(gw_sum):
    #     raise Exception('(gw_sum == 0.0) or ~isfinite(gw_temp))')
    
    return gw_temp/gw_sum, ind

def mp_find_nbr_qhull(xpos, ypos, npix, sm_num = 100, a = 1.0, b = 0.7, c = 1.0, print_space = 10000., nCores=cpu_count()):
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
    '''
    from scipy import spatial
    #The surface fitting performs better if the data is scattered about zero
    
    npix    = np.sqrt(npix)
    
    x0  = (xpos - np.median(xpos))/a
    y0  = (ypos - np.median(ypos))/b
    
    if np.sum(npix) != 0.0 and c != 0:
        np0 = (npix - np.median(npix))/c
    else:
        if np.sum(npix) == 0.0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because Noise Pixels are Zero')
        if c == 0:
            print('SKIPPING Noise Pixel Sections of Gaussian Kernel because c == 0')
    
    k            = sm_num                           # This is the number of nearest neighbors you want
    n            = x0.size                          # This is the number of data points you have
    nearest      = np.zeros((k,n),dtype=np.int64)   # This stores the nearest neighbors for each data point
    gw           = np.zeros((k,n),dtype=np.float64) # This is the gaussian weight for each data point determined from the nearest neighbors
    
    neigh = NearestNeighbors(n_neighbors=sm_num) # need 1 more to exclude the test point itself
    
    #Multiplying by 1000.0 avoids precision problems
    if npix.sum() != 0.0 and c != 0:
        # kdtree  = cKDTree(np.transpose((y0*1000., x0*1000., np0*1000.)))
        neigh.fit(np.transpose((y0*1000., x0*1000., np0*1000.)))
    else:
        # kdtree  = cKDTree(np.transpose((y0*1000., x0*1000.)))
        neigh.fit(np.transpose((y0*1000., x0*1000.)))
    
    inds = neigh.kneighbors(return_distance=False)
    
    func = partial(find_qhull_one_point, x0=x0, y0=y0, np0=np0, inds=inds, sm_num=sm_num)
    
    pool = Pool(nCores)
    
    gw_and_nbr = pool.starmap(func, zip(range(n)))
    
    pool.close()
    pool.join()
    
    gw_out = []
    nbr_out= []
    for gw_now, nbr_now in gw_and_nbr:
        gw_out.append(gw_now)
        nbr_out.append(nbr_now)
    
    return gw_out, nbr_out

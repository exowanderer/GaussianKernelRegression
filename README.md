# Gaussian Kernel Regression with k-Nearest Neighbors

This is a stand alone code that trains a 3D Gaussian kernel regression over the source positions and PSF widths (i.e. "beta pixels").

To implement in your own code, place the `gaussian_kernel_regression_3D.py` file in your working directory and run the `gaussian_weights_and_nearest_neighbors` function.

# Installation 
```bash
git clone https://github.com/exowanderer/GaussianKernelRegression
cd GaussianKernelRegression
python setup.py install
```

### Or

```bash
python3 -m pip install git+https://github.com/exowanderer/GaussianKernelRegression
```

# Example

### Create synthetic data
```python
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from gaussian_kernel_regression import gaussian_kernel_regression as gkr

nPts = int(1e5)

xpos = 0.35*np.sin(np.arange(0,nPts) / 1500 + 0.5) + 15 + np.random.normal(0,0.2,nPts)
ypos = 0.35*np.sin(np.arange(0,nPts) / 2000 + 0.7) + 15 + np.random.normal(0,0.2,nPts)
npix = 0.25*np.sin(np.arange(0,nPts) / 2500 + 0.4) + 15 + np.random.normal(0,0.2,nPts)
flux = 1+0.01*(xpos - xpos.mean()) + 0.01*(ypos - ypos.mean()) + 0.01*(npix - npix.mean())
```

### Train KDTree
```python
n_nbr   = 50  # Number of data points to sample inside cKDTree
points  = np.transpose([xpos,ypos,npix])  # set of coordinates to train cKDTree
kdtree  = cKDTree(points)  # trained KDtree
```

### Sample KDTree and calculate weights
```python
# Sample KDTree for list of nearby points per source frame
ind_kdtree  = kdtree.query(kdtree.data, n_nbr+1)[1][:,1:] # skip the first one because it's the current point

# Produce GKR weights over time
# `gaussian_kernel_regression` (i.e. `gkr`) only returns the gaussian weights in the indices are provided
gw_kdtree   = gkr.gaussian_weights(xpos, ypos, npix, ind_kdtree)
gkr_kdtree  = sum(flux[ind_kdtree] * gw_kdtree, axis=1)
```

### Plot the solution
```python
# Plot solution
fig1, ax1 = plt.subplots(1,1)
ax1.plot(flux, '.', ms=1, alpha=0.5)
ax1.plot(gkr_kdtree, '.', ms=1, alpha=0.5)

fig2, ax2 = plt.subplots(1,1)
ax2.plot(flux - gkr_kdtree  , '.', ms=1, alpha=0.5)
ax2.set_title('Scipy.cKDTree Gaussian Kernel Regression')
ax2.set_ylim(-0.0005,0.0005)
```

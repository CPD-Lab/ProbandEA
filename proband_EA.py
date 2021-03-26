# Copyright (c) 2020 Cognitive & Perceptual Developmental Lab
#                    Washington University School of Medicine

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Primary Contact: Muhamed Talovic
# Email: muhamed.talovic@wustl.edu

# Secondary Contact: Alexandre Todorov
# Email: todorov@wustl.edu

# The following functions have been vetted:
# percentThresh: (AT checked 05/11/2020)
# spearmanShuffle: (MT checked )
# (Maximum difference between our spearman function and scipy.stats.spearmanr is 7.7716E-16)


# --------------------------------- #
# Import necessary built-in functions

import numpy as np
from scipy.stats import rankdata


# --------------------------------- #
# Spearman (from Linear Models package - entire code not yet available to the public)

def spearmanShuffle(nshuffles, Y, F):
    # ----------------------------------------------------------------------
    # return the values with complete pheno and fc data
    # y = 0 int on error
    # ----------------------------------------------------------------------
    npairs, nobs = F.shape
    if len(Y) != nobs:
        print("ERROR: Dimensions of Y and F do not match")
        return None
    # ----------------------------------------------------------------------
    # Transform to ranks. Y is likely to have ties.
    # ----------------------------------------------------------------------
    W = rankdata(Y)
    meanW = np.mean(W)
    stdW = np.std(W)

    # ----------------------------------------------------------------------
    # Take advantage of the fact that since using ranks, the mean and the
    # variance of F remain virtually constant.
    # ----------------------------------------------------------------------
    V = np.array(range(1, nobs+1))
    R = np.zeros((npairs, nobs), dtype=int)
    Q = np.argsort(F, axis=1)
    for p in range(0, npairs):
        R[p, Q[p, :]] = V
    meanR = np.mean(R[0])
    stdR = np.std(R[0])

    # ----------------------------------------------------------------------
    # Generate shuffles, WS will be (shuffles+1)*nobs
    # ----------------------------------------------------------------------
    WS = np.array([W, ] * (nshuffles + 1))
    for i in range(1, nshuffles + 1):
        np.random.shuffle(WS[i, :])
    # ----------------------------------------------------------------------
    # Calculate correlations. Want simulations with replicates as rows
    # ----------------------------------------------------------------------
    acc2 = 1.0 / (float(nobs) * stdW * stdR)
    acc1 = nobs * meanW * meanR * acc2
    M = WS.dot(R.T) * acc2 - acc1
    return M

# --------------------------------- #
# Percent Threshold (from utilities package - entire code not yet available to the public)

def percentThresh(rho, p, sides, absval=False):
    """
    percentThresh(rho, p, sides, absval=False)
    Binarize a matrix using percentiles calculated for each row.
    Note: "U, 5" = upper 5% (i.e., the 95th percentile)
    L: B[i, j] = R[i, j] < t[i] where t[i] = percentile p, row i,
    U: B[i, j] = R[i, j] > t[i] where t[i] = percentile p, row i
    T: B[i, j] = R[i, j] > t1[i] OR R[i, j] < t2[i] where t1 and t2 are the p/2 and 100-p/2 percentiles

    :param rho: Input matrix
    :param p: percentile (1 < p <= 50)
    :param sides: L (lower), U (upper), T (2-sided)
    :param absval: Absolute value before calculating percentiles (default=False)
    :return: The binarized matrix (integer, 0 or 1)
    """
    if absval:
        rho = np.abs(rho)

    sides = sides.upper()
    if sides not in ['T', 'L', 'U']:
        print("ERROR Percentile invalid parameter")
        return None

    if p >= 50:
        print("ERROR Percentile must be less than 50")
        return None

    if sides == 'T':
        p /= 2.

    if rho.ndim == 1:
        if sides == 'L' or sides == 'T':
            t1 = np.percentile(rho, p)
        if sides == 'U' or sides == 'T':
            t2 = np.percentile(rho, 100-p)
    else:
        if sides == 'L' or sides == 'T':
            t1 = np.percentile(rho, p, axis=1).reshape(-1, 1)
        if sides == 'U' or sides == 'T':
            t2 = np.percentile(rho, 100-p, axis=1).reshape(-1, 1)

    if sides == 'U':
        F = (rho > t2).astype(int)
    elif sides == 'L':
        F = (rho < t1).astype(int)
    else:
        F = (rho < t1).astype(int) + (rho > t2).astype(int)
    return F


# ---------------------------------------------------------------------------- #
# Run analysis -- Entire code is not yet available to the public.

# I. Import input data and assign to ROI map -- not shown
# (code not yet available for public use):

n_shuffles = 50000 # (50,000 replicates)
alpha_level = 0.05 # (5% of hits)

Y = 'bx scores'
fc = 'correlation z-scores'

# -------------------------------------------------------------------------#
# II. RUNNING ENRICHMENT- SPEARMAN SHUFFLE (EMPIRICAL):

# 1. run spearman shuffle:
spearman_rhos_shuf = spearmanShuffle(n_shuffles, Y, fc)
#   returns matrix of rho values for each roi-roi pair
#   first the actual, and then the permuted/shuffled results
#   (shuffling the pairings of bx scores (Y) w/ correlation z-scores (fc))

spearman_hits_shuf = percentThresh(spearman_rhos_shuf, alpha_level * 100, 'T')
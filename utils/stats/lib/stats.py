# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
A replacement stats interface.

Replacement for 'f2py' built on stats.f90
This was motivated by difficulty compiling
stats.f90 properly. Should replace the
stats.f90 interface (simply not necessary)

author: Kyle Eskridge (GitHub:bkesk)
"""
import numpy as np
from numba import jit

@jit(nopython=True) 
def stddev(values): # pragma: no cover
    '''
    compute standard deviation as: 1/(N-1) * sum_i^N abs(values[i] - mean(values))**2

    note: this is equivalent to np.std(values, ddof=1) which does not compile in Numba
            (of course, np.std(values) does compile in Numba)
    '''
    n = len(values)
    mean = np.mean(values)
    s = 0
    for i in range(n):
        s+=(values[i] - mean)**2
    return np.sqrt(s/(n-1))


@jit(nopython=True)
def corr(values:np.ndarray): # pragma: no cover
    '''
    compute the correlation length of the input values

    correlation length, l, of values a(i) is given by:

    l = 1 +  sum_{1}^n ( 
        sum_{1}^{n-k} ( t(a,i,k) )  
        )

    where t(a,i,k) = T(a,i,k) if T(a,i,k)  > 0.0
                   = 0.0 otherwise,
        and
       T(a,i,k) = frac{2}{sigma^2} * (a(i) - mean(a)) * (a(i+k) - mean(a))
    '''

    mean_value = np.mean(values)
    sigma = stddev(values)

    if np.isclose(sigma,0.0):
        return np.inf
    n = values.shape[0]
    corr_len = 0.0
    for k in range(1,n+1):
        ct = 0
        for i in range(1,n-k+1):
            ct += (values[i-1] - mean_value)*(values[(i-1) + k] - mean_value)
        ct = (ct/sigma**2)/(n-k)
        if (ct <= 0.0):
            break
        corr_len = corr_len + 2*ct

    return corr_len + 1.0

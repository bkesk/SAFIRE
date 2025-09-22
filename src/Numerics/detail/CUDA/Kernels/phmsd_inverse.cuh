#ifndef PHMSD_INVERSE_H
#define PHMSD_INVERSE_H

#include <complex>

namespace kernels
{

void phmsd_inv(int nwalk,
               int ndet,
               int nex,
               int const* iexcit,
               std::complex<double> const* T,
               int ldT,
               long Tstride,
//               std::complex<double> const* ov, 
//               int ldo,
               std::complex<double>* Minv);

}
#endif

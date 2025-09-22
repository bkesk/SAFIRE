#ifndef PHMSD_DETERMINANTS_H
#define PHMSD_DETERMINANTS_H

#include <complex>

namespace kernels
{

void phmsd_det(int nwalk,
               int ndet,
               int nex,
               int const* iexcit,
               std::complex<double> const* T,
               int ldT,
               long Tstride,
               std::complex<double>* ov,
               int ldo);

}
#endif

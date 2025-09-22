#ifndef CONSTRUCT_PHMSD_R_H
#define CONSTRUCT_PHMSD_R_H
#include <complex>

namespace kernels
{
template<typename T1, typename T2, typename T3>
void construct_phmsd_R(int nwalk,
                       int ndet,
                       int nex,
                       int nact,
                       int nelec,
                       int const* iexcit,
                       int const* orbs,
                       std::complex<T1> const* T,
                       int ldT,
                       long Tstride,
                       std::complex<T2> const* I,
                       std::complex<T3>* Rbuff);

void reduce_phmsd_R(int nwalk, 
                    int ndet,
                    int nex,
                    int nact,
                    int nelec,
                    int const* iexcit,
                    int const* orbs,
                    std::complex<double> const* weights,
                    long ldw,
                    std::complex<double>* Rbuff,
                    std::complex<double>* R);


}
#endif

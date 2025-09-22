#ifndef PHMSD_ENERGY_H
#define PHMSD_ENERGY_H

#include <complex>

namespace kernels
{

template<typename T1, typename T2, typename T3>
void ph_excited_2body_energy_dense_cholesky_Tpna(int nwalk, int ndet, int nex, int nact, int nelec,
        int nchol, int const* iexcit, int const* refc, std::complex<T1> const* T,
        std::complex<T1> const* R, std::complex<T2> const* wgt, int ldW,
        std::complex<T3> *EX, int ldEX,
        std::complex<T3> *EJ, int ldEJ,
        std::complex<T1> *KE, int ldKE, long KEstride);

template<typename T1, typename T2, typename T3, typename T4>
void ph_excited_1body_energy(int nwalk, int ndet, int nex, int nact, int nelec, 
        int const* iexcit, int const* refc, std::complex<T1> const* S,
        std::complex<T2> const* R, std::complex<T3> const* wgt, int ldW,
        std::complex<T4> *E, int ldE);

}
#endif

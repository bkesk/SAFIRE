#ifndef EXTRACT_OVERLAP_MATRIX
#define EXTRACT_OVERLAP_MATRIX

#include <complex>

namespace kernels
{

void extract_overlap_matrix(int nwalk,
                    int ndet,
                    int nex,
                    int const* iexcit,
                    std::complex<double> const* T,
                    int ldT,
                    long Tstride,
                    std::complex<double> *M,
                    bool reverse_order = false);

}
#endif

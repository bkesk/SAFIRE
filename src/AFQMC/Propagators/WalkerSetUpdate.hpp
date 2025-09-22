////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#ifndef SFQMC_AFQMC_WALKERSETUPDATES_HPP
#define SFQMC_AFQMC_WALKERSETUPDATES_HPP

#include <algorithm>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "AFQMC/config.h"
#include "Numerics/ma_blas.hpp"
#include "AFQMC/Walkers/WalkerConfig.hpp"

namespace sfqmc
{
namespace afqmc
{
template<class Wlk, class OMat, class Mat, class WMat>
void free_projection_walker_update(Wlk&& w,
                                   RealType dt,
                                   OMat&& overlap,
                                   Mat&& MFfactor,
                                   RealType Eshift,
                                   Mat&& hybrid_weight,
                                   WMat& work,
                                   bool debug_verbosity)
{
  int nwalk = w.size();
  // constexpr if can be used to avoid the memory copy, by comparing the pointer types
  // between WMat and Mat/OMat
  if (work.size(0) < 7 || work.size(1) < nwalk)
    work = WMat({7, nwalk}, work.get_allocator());

  w.getProperty(WEIGHT, work[0]);
  w.getProperty(PHASE, work[1]);
  w.getProperty(PSEUDO_ELOC_, work[2]);
  w.getProperty(OVLP, work[3]);
  using std::copy_n;
  copy_n(overlap.origin(), nwalk, work[4].origin());
  copy_n(MFfactor.origin(), nwalk, work[5].origin());
  copy_n(hybrid_weight.origin(), nwalk, work[6].origin());

  for (int i = 0; i < nwalk; i++)
  {
    ComplexType old_ovlp = work[3][i];
    ComplexType old_eloc = work[2][i];
    ComplexType eloc;
    ComplexType ratioOverlaps = ComplexType(1.0, 0.0);
    eloc                      = work[5][i] / dt;
    ComplexType factor        = std::exp(-dt * (0.5 * (eloc + old_eloc) - Eshift));
  
    if (debug_verbosity)
    {
      std::cout << " update: iw:       " <<i <<"\n"     
        << "    eloc:          " << eloc << "\n"
        << "    ov:            " << work[4][i] << "\n"
        << "    old_ov:        " << old_ovlp << "\n"
        << "    old_eloc:      " << old_eloc<< "\n"
        << "    old_weight:    " << work[0][i] << "\n"
        << "    ratio:         " << ratioOverlaps << "\n"
        << "    MFfactor:      " << work[5][i] << "\n"
        << "    hybrid_weight: " << work[6][i] << "\n"
        << "    Eshift:         " << Eshift << "\n"
        << "    factor:         " << factor << "\n"
        << std::endl;
    }
    
    work[0][i] *= std::abs(factor);
    work[1][i] *= factor / std::abs(factor);
    work[2][i] = eloc;
    work[3][i] = work[4][i];
  }

  w.setProperty(WEIGHT, work[0]);
  w.setProperty(PHASE, work[1]);
  w.setProperty(PSEUDO_ELOC_, work[2]);
  w.setProperty(OVLP, work[3]);
    
}


template<class Wlk, class OMat, class Mat, class WMat>
void hybrid_walker_update(Wlk&& w,
                          RealType dt,
                          bool apply_constrain,
                          bool imp_sampl,
                          RealType Eshift,
                          OMat&& overlap,
                          Mat&& MFfactor,
                          Mat&& hybrid_weight,
                          WMat& work,
			  double lower_cutoff_scale,
			  double upper_cutoff_scale,
			  bool symmetric_split)
{
    hybrid_walker_update(
      w, 
      dt, 
      apply_constrain, 
      imp_sampl, 
      Eshift, 
      overlap, 
      MFfactor, 
      hybrid_weight,
      work, 
      lower_cutoff_scale, 
      upper_cutoff_scale, 
      symmetric_split, 
      false, // debug_verbosity 
      false  // use_cp_constraint
      );
}

template<class Wlk, class OMat, class Mat, class WMat>
void hybrid_walker_update(Wlk&& w,
                          RealType dt,
                          bool apply_constrain,
                          bool imp_sampl,
                          RealType Eshift,
                          OMat&& overlap,
                          Mat&& MFfactor,
                          Mat&& hybrid_weight,
                          WMat& work,
			  double lower_cutoff_scale,
			  double upper_cutoff_scale,
			  bool symmetric_split,
        bool debug_verbosity) // KE: debug_verbosity may be used in any propagtor, use_cp_constraint is specific to models
{                             //    so debug_verbosity should be bfore use_cp_constraint in the parameter list
    hybrid_walker_update(
      w, 
      dt, 
      apply_constrain, 
      imp_sampl, 
      Eshift, 
      overlap, 
      MFfactor, 
      hybrid_weight,
      work, 
      lower_cutoff_scale, 
      upper_cutoff_scale, 
      symmetric_split, 
      debug_verbosity,
      false // use_cp_constraint
      );
}

template<class Wlk, class OMat, class Mat, class WMat>
void hybrid_walker_update(Wlk&& w,
                          RealType dt,
                          bool apply_constrain,
                          bool imp_sampl,
                          RealType Eshift,
                          OMat&& overlap,
                          Mat&& MFfactor,
                          Mat&& hybrid_weight,
                          WMat& work,
                          double lower_cutoff_scale,
                          double upper_cutoff_scale,
                          bool symmetric_split,
                          bool debug_verbosity,
                          bool use_cp_constraint)
{
  int nwalk = w.size();
  // constexpr if can be used to avoid the memory copy, by comparing the pointer types
  // between WMat and Mat/OMat
  if (work.size(0) < 11 || work.size(1) < nwalk)
    work = WMat({11, nwalk}, work.get_allocator());

  bool BackProp = (w.getBPPos() >= 0 && w.getBPPos() < w.NumBackProp());

  w.getProperty(WEIGHT, work[0]);
  w.getProperty(PSEUDO_ELOC_, work[1]);
  w.getProperty(OVLP, work[2]);
  w.getProperty(PHASE1, work[7]);
  w.getProperty(PHASE2, work[8]);
  w.getProperty(PHASE3, work[9]);
  using std::copy_n;
  copy_n(overlap.origin(), nwalk, work[4].origin());
  copy_n(MFfactor.origin(), nwalk, work[5].origin());
  copy_n(hybrid_weight.origin(), nwalk, work[6].origin());

  for (int i = 0; i < nwalk; i++)
  {
    ComplexType old_ovlp = work[2][i];
    ComplexType old_eloc = work[1][i];
    ComplexType eloc;
    RealType delta_theta;
    RealType scale            = 1.0;
    ComplexType ratioOverlaps = ComplexType(1.0, 0.0);

    if (imp_sampl)
    {
      auto dbg_tmp1 = work[4][i];
      ratioOverlaps = dbg_tmp1 / old_ovlp;
    }

    ComplexType ratioOverlaps_ = ratioOverlaps;
    
    if (!std::isfinite(ratioOverlaps.real()) && apply_constrain && imp_sampl)
    {
      scale = 0.0;
      eloc  = old_eloc;
    }
    else
    {
      // save constraint theta for sanity checks
      delta_theta = std::arg(ratioOverlaps) - work[5][i].imag();
      work[10][i] = delta_theta;
      if (use_cp_constraint)
      {
        // if real part of ratioOverlaps is positive, scale is 1.0 otherwise, scale is 0.0
        scale = (std::cos(delta_theta) > 0.0 ? 1.0 : 0.0);
        ratioOverlaps = std::real(ratioOverlaps); // is this needed?
      }
      else 
      {
        scale = (apply_constrain ? std::max(0.0, std::cos(delta_theta)) : 1.0);
      }
      
      if (imp_sampl)
        eloc = (work[5][i] - work[6][i] - std::log(ratioOverlaps_)) / dt;
      else
        eloc = work[5][i] / dt;
    }
    ComplexType eloc_ = eloc;

    if ((!std::isfinite(eloc.real())) || (std::abs(eloc.real()) < std::numeric_limits<RealType>::min()))
    {
      scale = 0.0;
      eloc  = old_eloc;
    }
    else
    {
	eloc = ComplexType( std::max( std::min(eloc.real(), 
			Eshift + upper_cutoff_scale * std::sqrt(2.0 / dt)), 
			Eshift - lower_cutoff_scale * std::sqrt(2.0 / dt)), eloc.imag());
    }
    
    if (debug_verbosity)
    {
    std::cout << " update: iw:       " <<i <<"\n"     
              << "    eloc:          " << eloc << "\n"
              << "    eloc_:         " << eloc_ << "\n"
              << "    ov:            " << work[4][i] << "\n"
              << "    old_ov:        " << old_ovlp << "\n"
              << "    old_eloc:      " << old_eloc<< "\n"
              << "    old_weight:    " << work[0][i] << "\n"
              << "    ratio:         " << ratioOverlaps << "\n"
              << "    MFfactor:      " << work[5][i] << "\n"
              << "    hybrid_weight: " << work[6][i] << "\n"
              << "    scale:         " << scale << "\n"
              << "    Eshift:         " << Eshift << "\n"
              << "    Theta:         " << work[10][i] << "\n"
              << std::endl;
    }

    if(symmetric_split)
      work[0][i] *= ComplexType(scale * std::exp(-dt * (0.5 * (eloc.real() + old_eloc.real()) - Eshift)), 0.0);
    else
      work[0][i] *= ComplexType(scale * std::exp(-dt * (eloc.real() - Eshift)), 0.0);
    work[1][i] = eloc;
    work[2][i] = work[4][i];
    if (scale !=0)
      work[3][i] = std::exp(-ComplexType(0.0, dt) * (0.5 * (eloc.imag() + old_eloc.imag()))) / scale;
    else
      work[3][i] = 0.0;
    work[7][i] *= work[3][i]; 
    work[8][i] = work[4][i]; 
    work[9][i] = scale; // KE: this was originally the cumulative product of "scale"
                        //    changed to just "scale" since this isn't used anywhere 
  }
  w.setProperty(WEIGHT, work[0]);
  w.setProperty(PSEUDO_ELOC_, work[1]);
  w.setProperty(OVLP, work[2]);
  w.setProperty(PHASE1, work[7]);
  w.setProperty(PHASE2, work[8]);
  w.setProperty(PHASE3, work[9]);
  w.setProperty(THETA, work[10]);
  if (BackProp)
  {
    auto&& WFac((*w.getWeightFactors())[w.getHistoryPos()]);
    using std::copy_n;
    copy_n(work[3].origin(), nwalk, WFac.origin());
    auto&& WHis((*w.getWeightHistory())[w.getHistoryPos()]);
    copy_n(work[0].origin(), nwalk, WHis.origin());
  }
}

template<class Wlk, class EMat, class OMat, class Mat, class WMat>
void local_energy_walker_update(Wlk&& w,
                                RealType dt,
                                bool apply_constrain,
                                RealType Eshift,
                                OMat&& overlap,
                                EMat&& energies,
                                Mat&& MFfactor,
                                [[maybe_unused]] Mat&& hybrid_weight,
                                WMat& work,
				double lower_cutoff_scale,
				double upper_cutoff_scale)
{
  int nwalk = w.size();
  // constexpr if can be used to avoid the memory copy, by comparing the pointer types
  // between WMat and Mat/OMat
  if (work.size(0) < 14 || work.size(1) < nwalk)
    work = WMat({14, nwalk}, work.get_allocator());

  bool BackProp = (w.getBPPos() >= 0 && w.getBPPos() < w.NumBackProp());

  w.getProperty(WEIGHT, work[0]);
  w.getProperty(PSEUDO_ELOC_, work[1]);
  w.getProperty(OVLP, work[2]);
  w.getProperty(E1_, work[3]);
  w.getProperty(EXX_, work[4]);
  w.getProperty(EJ_, work[5]);
  w.getProperty(PHASE, work[12]);
  using std::copy_n;
  copy_n(overlap.origin(), nwalk, work[7].origin());
  copy_n(MFfactor.origin(), nwalk, work[8].origin());
  ma::copy(energies({0, nwalk}, 0), work[9]);
  ma::copy(energies({0, nwalk}, 1), work[10]);
  ma::copy(energies({0, nwalk}, 2), work[11]);

  for (int i = 0; i < nwalk; i++)
  {
    ComplexType old_ovlp      = work[2][i];
    ComplexType old_eloc      = work[1][i];
    ComplexType eloc          = work[9][i] + work[10][i] + work[11][i];
    RealType scale            = 1.0;
    ComplexType ratioOverlaps = work[7][i] / old_ovlp;

    if (!std::isfinite((ratioOverlaps * work[8][i]).real()) && apply_constrain)
    {
      scale = 0.0;
      eloc  = old_eloc;
    }
    else
    { 
      work[13][i] = std::arg(ratioOverlaps) - work[8][i].imag();
      scale = (apply_constrain ? (std::max(0.0, std::cos(std::arg(ratioOverlaps) - work[8][i].imag()))) : 1.0);
    }
    if ((!std::isfinite(eloc.real())) || (std::abs(eloc.real()) < std::numeric_limits<RealType>::min()))
    {
      scale = 0.0;
      eloc  = old_eloc;
    }
    else
    {
      eloc = ComplexType( std::max( std::min(eloc.real(), 
			Eshift + upper_cutoff_scale * std::sqrt(2.0 / dt)), 
			Eshift - lower_cutoff_scale * std::sqrt(2.0 / dt)), eloc.imag());
    }

    work[0][i] *= ComplexType(scale * std::exp(-dt * (0.5 * (eloc.real() + old_eloc.real()) - Eshift)), 0.0);
    work[6][i] = std::exp(-ComplexType(0.0, dt) * (0.5 * (eloc.imag() + old_eloc.imag()))) / scale;
    work[12][i] *= work[6][i]; 
    work[1][i] = eloc;
    work[2][i] = work[7][i];
    work[3][i] = work[9][i];
    work[4][i] = work[10][i];
    work[5][i] = work[11][i];
  }

  w.setProperty(WEIGHT, work[0]);
  w.setProperty(PSEUDO_ELOC_, work[1]);
  w.setProperty(OVLP, work[2]);
  w.setProperty(E1_, work[3]);
  w.setProperty(EXX_, work[4]);
  w.setProperty(EJ_, work[5]);
  w.setProperty(PHASE, work[12]);
  w.setProperty(THETA, work[13]);
  if (BackProp)
  {
    auto&& WFac((*w.getWeightFactors())[w.getHistoryPos()]);
    using std::copy_n;
    copy_n(work[6].origin(), nwalk, WFac.origin());
    auto&& WHis((*w.getWeightHistory())[w.getHistoryPos()]);
    copy_n(work[0].origin(), nwalk, WHis.origin());
  }
}

} // namespace afqmc

} // namespace sfqmc

#endif

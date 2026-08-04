#pragma once

#include <string>
#include <vector>
#include <optional>

#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "utilities/json.hpp"


namespace sfqmc::afqmc {

enum class DriverType {
  afqmc,
  ftafqmc,
};
SAFIRE_DEFINE_ENUM_NAMES(DriverType, afqmc, ftafqmc);

SAFIRE_DEFINE_ENUM(WALKER_TYPES, {
  {UNDEFINED_WALKER_TYPE, "undefined"},
  {CLOSED, "closed"},
  {COLLINEAR, "collinear"},
  {NONCOLLINEAR, "noncollinear"},
});

SAFIRE_DEFINE_ENUM_NAMES(LoadBalanceAlgorithm, undefined, simple, async);
SAFIRE_DEFINE_ENUM_NAMES(BranchingAlgorithm, undefined, pair, comb, min_branch, serial_comb);

  
enum class PHMSDEnergyAlgorithm {
  reference,
  woodbury,
};
SAFIRE_DEFINE_ENUM_NAMES(PHMSDEnergyAlgorithm, reference, woodbury);


enum class EstimatorType {
  undefined,
  basic,
  mixed,
  energy,
  back_propagation,
  time_evolved_operators,
};
SAFIRE_DEFINE_ENUM_NAMES(EstimatorType, undefined, basic, mixed, energy, back_propagation, time_evolved_operators);


struct ProjectParameters {
  std::string id{"afqmc"};
  int series{};
};
SAFIRE_DEFINE_PARAMETERS(ProjectParameters, id, series);

struct WalkerSetParameters {
  std::string name{"wset0"};
  WALKER_TYPES walker_type{COLLINEAR};
  LoadBalanceAlgorithm load_balance_type{LoadBalanceAlgorithm::async};
  BranchingAlgorithm pop_control_type{BranchingAlgorithm::pair};
  double min_weight{0.05};
  double max_weight{4.0};
  bool finite_temperature{};
};
SAFIRE_DEFINE_PARAMETERS(WalkerSetParameters, name, walker_type, load_balance_type, pop_control_type, min_weight,
                         max_weight, finite_temperature);


struct WavefunctionParameters {
  std::string name{};
  std::string filename{}; // required

  bool rediag{}; // ??
  int ndets_to_read{-1};
  // TODO: default depends on the hamiltonian type in the input file (woodbury for RealDenseFactorized,
  // reference otherwise)
  std::optional<PHMSDEnergyAlgorithm> algorithm{};
  // TODO: default depends on the hamiltonian type in the input file (false for KPFactorized and KPTHC,
  // true otherwise)
  std::optional<bool> dense_trial{};
  int nwalk_block_size{8};
  int ndet_block_size{4096};

  // system
};
SAFIRE_DEFINE_PARAMETERS(WavefunctionParameters, name, filename, rediag, ndets_to_read, algorithm, dense_trial,
                         nwalk_block_size, ndet_block_size);

struct HamiltonianParameters {
  std::string name{"ham0"};
  std::string filename{}; // TODO: defaults to the filename of the wavefunction block
  int max_memory{2000};   // MiB
  bool shift_1body{};
  int buffer_size{4096};
};
SAFIRE_DEFINE_PARAMETERS(HamiltonianParameters, name, filename, max_memory, shift_1body, buffer_size);

struct PropagatorParameters {
  // TODO: for a ModelHamiltonian input file the defaults change to vbias_bound = 100.0,
  // upper_cutoff_scale = lower_cutoff_scale = 50.0, denseP2 = false and symmetric_split = false
  double taylor_n{6};
  double vbias_bound{50.0};
  double external_field_scale{1.0};
  double upper_cutoff_scale{10.0};
  double lower_cutoff_scale{1.0};
  bool apply_constrain{true};
  bool importance_sampling{true};
  bool substractMF{true};
  bool hybrid{true};
  bool printP1eigval{false};
  bool free_projection{false};
  bool denseP1{false};
  bool denseP2{true};
  bool debug_verbosity{false};
  bool natural_shift{true};
  bool symmetric_split{true};
  bool use_cp_constraint{false};
  bool use_real_vbias{false};
  std::string external_field{""};
  std::string excited{""};
};
SAFIRE_DEFINE_PARAMETERS(PropagatorParameters, taylor_n, vbias_bound, external_field_scale, upper_cutoff_scale,
                         lower_cutoff_scale, apply_constrain, importance_sampling, substractMF, hybrid,
                         printP1eigval, free_projection, denseP1, denseP2, debug_verbosity, natural_shift,
                         symmetric_split, use_cp_constraint, use_real_vbias, external_field, excited);

struct OneRDMParameters {
  std::string rotation{};
  std::string path{"/"};
  bool with_index_list{false};
};
SAFIRE_DEFINE_PARAMETERS(OneRDMParameters, rotation, path, with_index_list);

struct DiagTwoRDMParameters {
};
SAFIRE_DEFINE_EMPTY_PARAMETERS(DiagTwoRDMParameters);

struct TwoRDMParameters {
  std::string rotation{};
  std::string path{"/"};
};
SAFIRE_DEFINE_PARAMETERS(TwoRDMParameters, rotation, path);

struct PairCorrelatorParameters {
    std::string name{"pair_correlator"};
    std::string walker_output{""};
    std::string filename{""};

    std::vector<std::string> pair_type{}; // required, at least one entry
};
SAFIRE_DEFINE_PARAMETERS(PairCorrelatorParameters, name, walker_output, filename, pair_type);

struct SpinSpinCorrParameters {
};
SAFIRE_DEFINE_EMPTY_PARAMETERS(SpinSpinCorrParameters);

struct EstimatorParameters {
  EstimatorType name{};
  bool remove{false};

  // the estimator may use a different wavefunction and hamiltonian than the driver
  std::string wfn{};
  std::string ham{};

  // basic
  bool timers{false};
  int nhist{0};

  // energy
  bool overwrite{false};
  bool print_components{};
  bool print_sign{};

  int equil{};
  int skip{}; // ?

  // mixed
  int equil_multiplier{};

  // bp
  int bp_walker_ortho_interval{10};  // TODO: defaults to 1 for name = time_evolved_operators
  bool path_restoration{true};
  bool extra_path_restoration{false}; // TODO: defaults to true for name = time_evolved_operators

  // TODO: defaults to the measure_interval_multiplier of the enclosing execute block
  std::vector<int> measure_interval_multiplier{{DEFAULT_MEASURE_INTERVAL_MULTIPLIER}};

  // observables
  std::optional<OneRDMParameters> onerdm{};
  std::optional<DiagTwoRDMParameters> diag2rdm{};
  std::optional<TwoRDMParameters> twordm{};
  std::optional<PairCorrelatorParameters> pair_correlators{};
  std::optional<SpinSpinCorrParameters> spinspin{};
};
SAFIRE_DEFINE_PARAMETERS(EstimatorParameters, name, remove, wfn, ham, timers, nhist, overwrite, print_components,
                         print_sign, equil, skip, equil_multiplier, bp_walker_ortho_interval, path_restoration,
                         extra_path_restoration, measure_interval_multiplier, onerdm, diag2rdm, twordm,
                         pair_correlators, spinspin);


struct ExecuteBlock {
  WalkerSetParameters walker_set{};
  WavefunctionParameters wavefunction{};
  HamiltonianParameters hamiltonian{};
  PropagatorParameters propagator{};
  std::vector<EstimatorParameters> estimator{};


  std::string hdf_read_file{}; // restart from checkpoint
  std::string hdf_write_file{}; // write checkpoint
  int steps{1};
  int sweeps{1}; // finite temperature sweeps
  int population_control_interval{DEFAULT_POPULATION_CONTROL_INTERVAL};
  int measure_interval_multiplier{DEFAULT_MEASURE_INTERVAL_MULTIPLIER};
  int walker_ortho_interval{DEFAULT_WALKER_ORTHO_INTERVAL};
  int checkpoint_interval{-1};
  double weight_reset{0.0}; // in units of time
  double dshift{1.0};
  bool print_sweep_step{false}; // ftafqmc only

  // fix_bias // CSAFQMC only
  // filename
  // ndets_to_read

  double timestep{DEFAULT_TIME_STEP};
  int n_walkers_per_mpi_task{10};
  bool set_nwalker_to_target{};
  double initial_Eshift{}; // make optional
  int seed{};
};
SAFIRE_DEFINE_PARAMETERS(ExecuteBlock, walker_set, wavefunction, hamiltonian, propagator, estimator, hdf_read_file,
                         hdf_write_file, steps, sweeps, population_control_interval, measure_interval_multiplier,
                         walker_ortho_interval, checkpoint_interval, weight_reset, dshift, print_sweep_step,
                         timestep, n_walkers_per_mpi_task, set_nwalker_to_target, initial_Eshift, seed);


struct AFQMCParameters {
  DriverType driver{DriverType::afqmc};

  ProjectParameters project{};

  std::vector<ExecuteBlock> execute{};

  // blocks declared outside of an execute block have to be named, so that an execute block can refer to them
  std::vector<WalkerSetParameters> walker_set{};
  std::vector<WavefunctionParameters> wavefunction{};
  std::vector<HamiltonianParameters> hamiltonian{};
  std::vector<PropagatorParameters> propagator{};
};
SAFIRE_DEFINE_PARAMETERS(AFQMCParameters, driver, project, execute, walker_set, wavefunction, hamiltonian,
                         propagator);



}

/*
 * This file is part of teralens, a GPU-based software for the
 * efficient calculation of magnification patterns due to gravitational
 * microlensing.
 *
 * Copyright (C) 2018  Aksel Alpay
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TERALENS_CONFIGURATION_HPP
#define TERALENS_CONFIGURATION_HPP

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>

#include <SpatialCL/configuration.hpp>

namespace teralens {

using scalar = float;

// Use a type system defined by the scalar type, in 2D, with 3 components per
// particle
using type_system = spatialcl::type_descriptor::generic<scalar, 2, 3>;

using vector16 = spatialcl::cl_vector_type<scalar,16>::value;
using vector8  = spatialcl::cl_vector_type<scalar, 8>::value;
using vector2  = spatialcl::cl_vector_type<scalar, 2>::value;


using particle_type =
  spatialcl::configuration<type_system>::particle_type;

using vector_type =
  spatialcl::configuration<type_system>::vector_type;

/// Whether the evaluation order of primary rays should be reordered
/// by grouping rays into tiles. This can reduce branch divergence
/// during the tree walk, and is generally recommended for execution
/// on GPUs.
#ifdef TERALENS_CPU_FALLBACK
static constexpr bool reorder_primary_rays = false;
#else
static constexpr bool reorder_primary_rays = true;
#endif


/// The OpenCL group size for the primary ray tree queries
#ifdef TERALENS_CPU_FALLBACK
// A group size of 0 means that a cl::NullRange will be used
// as local size, hence allowing the OpenCL implementation
// to select the group size
static constexpr std::size_t primary_ray_query_group_size = 32;
#else
static constexpr std::size_t primary_ray_query_group_size = 128;
#endif

// The OpenCL group size for the secondary ray tracing.
// secondary_rays_per_cell^2 must be a multiple of this number.
// Must be >= max_selected_particles, except for the CPU fallback
// if allow_local_mem_on_cpu == false.
#ifdef TERALENS_CPU_FALLBACK
  static constexpr std::size_t secondary_ray_tracing_group_size = 32;
#else
  static constexpr std::size_t secondary_ray_tracing_group_size = 128;
#endif

/// Whether fused multiply-add instructions should be used
static constexpr bool allow_fma_instructions = true;
/// Whether multiply-add instructions should be used
static constexpr bool allow_mad_instructions = true;
/// Wether to allow the usage of local memory for the
/// CPU backend. On some OpenCL implementations, local
/// memory seems to accelerate the computation even on CPUs
/// (Intel?). On others, it may slow down the computation. (pocl?)
static constexpr bool allow_local_mem_on_cpu = false;

/// Maximum number of lenses calculated with the brute force algorithm.
/// Usually, the tree backend is already faster for few dozen
/// lenses.
static constexpr std::size_t max_brute_force_lenses = 64;

/// The number of secondary rays that are evaluated per interpolation
/// cell. The total number of evaluated rays per primary ray
/// is \c num_interpolation_cells * secondary_rays_per_cells^2.
/// Must be a power of two, and secondary_rays_per_cell^2
/// must be a multiple of the warp size (32 for NVIDIA, 64
/// for AMD) and 128 (the work group size for the evaluation
/// of the lens equation).
/// Feel free to increase this for a cheap (in terms of runtime)
/// increase in Signal/Noise.
static constexpr std::size_t secondary_rays_per_cell = 32;
/// Maximum number of primary rays processed in one batch.
static constexpr std::size_t max_batch_size = 256*1024;
/// The maximum number of particles to be treated exactly for
/// each primary ray. Note that teralens will attempt to allocate
/// a buffer with a size of
/// \c max_selected_particles * max_batch_size, so do not
/// make this number too large! In practice, even 64 (or less?)
/// seems to suffice.
static constexpr std::size_t max_selected_particles = 128;

// Do not change - changing this will not actually change the number
// of cells!
static constexpr std::size_t num_interpolation_cells = 4;


}

#endif

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

// Must be a power of two, and secondary_rays_per_cell^2
// must be a multiple of the warp size (32 for NVIDIA, 64
// for AMD)
static constexpr std::size_t secondary_rays_per_cell = 16;

static constexpr std::size_t max_batch_size = 32*1024;

static constexpr std::size_t max_selected_nodes = 256;
static constexpr std::size_t max_selected_particles = 128;


}

#endif

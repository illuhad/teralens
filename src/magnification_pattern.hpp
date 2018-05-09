
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

#ifndef MAGNIFICATION_PATTERN_HPP
#define MAGNIFICATION_PATTERN_HPP

#include <cassert>
#include <vector>
#include <cmath>
#include <random>

#include <QCL/qcl_array.hpp>
#include <SpatialCL/query.hpp>

#include "configuration.hpp"
#include "lensing_tree.hpp"
#include "lensing_tree_walk.hpp"

namespace teralens {

class lensing_system
{
public:
  lensing_system(const qcl::device_context_ptr& ctx,
                 scalar mean_particle_mass,
                 scalar convergence_stars,
                 scalar convergence_smooth,
                 scalar shear,
                 scalar source_plane_size,
                 std::size_t random_seed = 12345,
                 scalar overshooting_region_size = 6.0f,
                 scalar lens_and_ray_region_size_ratio = 1.5f)
    : _convergence_c{convergence_stars},
      _convergence_s{convergence_smooth},
      _shear{shear},
      _mag_pattern_size{source_plane_size},
      _overshooting{overshooting_region_size},
      _lens_distribution_size_factor{lens_and_ray_region_size_ratio}
  {
    assert(_mag_pattern_size > 0.0f);
    assert(lens_and_ray_region_size_ratio > 1.0f);

    _stars = this->sample_particles(ctx, mean_particle_mass, random_seed);
  }

  scalar get_compact_convergence() const
  {
    return _convergence_c;
  }

  scalar get_smooth_convergence() const
  {
    return _convergence_s;
  }

  scalar get_total_convergence() const
  {
    return _convergence_c + _convergence_s;
  }

  scalar get_shear() const
  {
    return _shear;
  }

  vector2 get_shooting_region_extent() const
  {
    scalar kappa = this->get_total_convergence();
    scalar gamma = this->get_shear();

    vector2 result;
    result.s[0] = (_mag_pattern_size + _overshooting) / (1.0f - gamma - kappa);
    result.s[1] = (_mag_pattern_size + _overshooting) / (1.0f + gamma - kappa);

    return result;
  }

  const qcl::device_array<particle_type>& get_particles() const
  {
    return _stars;
  }

  scalar get_source_plane_size() const
  {
    return _mag_pattern_size;
  }

private:
  scalar _convergence_c;
  scalar _convergence_s;
  scalar _shear;
  scalar _mag_pattern_size;
  scalar _overshooting;
  scalar _lens_distribution_size_factor;
  qcl::device_array<particle_type> _stars;

  qcl::device_array<particle_type> sample_particles(const qcl::device_context_ptr& ctx,
                                                    scalar mean_particle_mass,
                                                    std::size_t seed) const
  {
    vector2 shooting_region = this->get_shooting_region_extent();
    scalar lens_radius =
        _lens_distribution_size_factor * 0.5f *
        std::sqrt(shooting_region.s[0]*shooting_region.s[0]
                + shooting_region.s[1]*shooting_region.s[1]);

    std::size_t num_particles =
        static_cast<std::size_t>(
          std::round(_convergence_c * lens_radius * lens_radius / mean_particle_mass));

    std::vector<particle_type> particles(num_particles);

    // ToDo: Sample particles on the GPU
    std::mt19937 random_engine(seed);
    std::uniform_real_distribution<scalar> random_distribution{-lens_radius, lens_radius};

    scalar lens_radius2 = lens_radius * lens_radius;
    for(std::size_t i = 0; i < num_particles; ++i)
    {
      particle_type p;
      p.s[2] = mean_particle_mass;

      do
      {
        p.s[0] = random_distribution(random_engine);
        p.s[1] = random_distribution(random_engine);
      } while(p.s[0]*p.s[0] + p.s[1]*p.s[1] > lens_radius2);

      particles[i] = p;
    }

    return qcl::device_array<particle_type>{ctx, particles};
  }
};

using grouped_dfs_query_engine = spatialcl::query::engine::grouped_depth_first
<
  lensing_tree,
  ray_grid_query
>;

using dfs_query_engine = spatialcl::query::engine::depth_first
<
  lensing_tree,
  ray_grid_query,
  spatialcl::query::engine::HIERARCHICAL_ITERATION_RELAXED
>;

class magnification_pattern_generator
{
public:

  magnification_pattern_generator(const qcl::device_context_ptr& ctx,
                                  const lensing_system& system)
    : _ctx{ctx},
      _system{system},
      _tree{ctx, system.get_particles()}
  {}

  qcl::device_array<int> run(std::size_t resolution,
                             std::size_t num_rays_ppx,
                             scalar tree_opening_angle = 0.4)
  {

    scalar ray_distance = _system.get_source_plane_size() /
        (std::sqrt(static_cast<scalar>(num_rays_ppx)) * resolution);

    vector2 primary_ray_separation;
    primary_ray_separation.s[0] = ray_distance;
    primary_ray_separation.s[1] = ray_distance;

    ray_grid_query lensing_query{
      _ctx,
      // shooting region center
      {{0.0f, 0.0f}},
      // ray grid extent
      _system.get_shooting_region_extent(),
      // distance between rays
      primary_ray_separation,
      // Center of the pixel screen in the source plane
      {{0.0f, 0.0f}},
      // Physical size of the magnification pattern
      _system.get_source_plane_size()/resolution,
      // Number of pixels in x-direction
      resolution,
      // Numver of pixels in y-direction
      resolution,
      // global shear
      _system.get_shear(),
      // Convergence due to smooth matter
      _system.get_smooth_convergence(),
      // opening angle of the tree walk -
      // use smaller values for a more exact
      // (but slower) computation
      tree_opening_angle
    };

    //using query_engine_type = grouped_dfs_query_engine;
    using query_engine_type = dfs_query_engine;

    query_engine_type lensing_query_engine;
    lensing_query_engine(this->_tree, lensing_query);

    qcl::check_cl_error(_ctx->get_command_queue().finish(),
                        "Error while waiting for the lensing query "
                        "to finish.");
    return lensing_query.get_pixel_screen();
  }

private:
  qcl::device_context_ptr _ctx;
  lensing_system _system;
  lensing_tree _tree;

};

}

#endif

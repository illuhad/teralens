
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
  primary_ray_query
>;

using dfs_query_engine = spatialcl::query::engine::depth_first
<
  lensing_tree,
  primary_ray_query,
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
                             scalar num_primary_rays_ppx,
                             scalar tree_opening_angle = 0.4)
  {
    using query_engine_type = dfs_query_engine;

    // Calculate lower-left corner of the shooting region
    vector2 shooting_region_min_corner = _system.get_shooting_region_extent();
    for(int i = 0; i < 2; ++i)
      shooting_region_min_corner.s[i] *= -0.5f;

    // Calculate distance between primary rays
    scalar px_size_x = _system.get_shooting_region_extent().s[0] / resolution;
    scalar px_size_y = _system.get_shooting_region_extent().s[1] / resolution;
    scalar ray_distance = std::sqrt(px_size_x * px_size_y / num_primary_rays_ppx);


    // Create a ray_scheduler object, which will be used to calculate
    // the coordinates of the primary rays
    ray_scheduler scheduler{
      _ctx,
      shooting_region_min_corner,
      ray_distance,
      static_cast<std::size_t>(_system.get_shooting_region_extent().s[0] / ray_distance),
      static_cast<std::size_t>(_system.get_shooting_region_extent().s[1] / ray_distance),
      max_batch_size
    };

    // Create pixel screen object, which will count the rays in each pixel
    pixel_screen screen{
      _ctx,
      resolution,
      resolution,
      {{0.0f, 0.0f}},
      {{_system.get_source_plane_size(), _system.get_source_plane_size()}}
    };


    query_engine_type lensing_query_engine;
    for(int i = 0; !scheduler.all_rays_processed(); ++i)
    {
      std::cout << "--> Tracing rays of batch " << i << std::endl;

      // Generate a batch of primary ray positions on the GPU
      std::size_t num_rays_in_batch = scheduler.generate_ray_batch();
      std::cout << "  Scheduled " << num_rays_in_batch << " primary rays." << std::endl;

      // Formulate tree query to determine which stars must be included
      // exactly in the calculation, and which nodes' multipole expansion
      // must be included.
      primary_ray_query ray_query{
        _ctx,
        scheduler.get_current_batch(),
        num_rays_in_batch,
        max_selected_particles,
        ray_distance,
        tree_opening_angle
      };
      // Run tree query
      std::cout << "  Running tree queries for primary rays..." << std::endl;
      lensing_query_engine(this->_tree, ray_query);

      // Create secondary ray tracer, which will create interpolation cells
      // for the long-range deflections and evaluate the close-range
      // deflections from close stars at many locations for each primary ray
      secondary_ray_tracer<max_selected_particles> ray_evaluator {
        _ctx,
        &screen
      };

      // Run secondary ray tracer
      std::cout << "  Tracing secondary rays..." << std::endl;
      ray_evaluator(scheduler.get_current_batch(),
                    num_rays_in_batch,
                    ray_distance,
                    ray_query.get_num_selected_particles(),
                    ray_query.get_selected_particles(),
                    ray_query.get_interpolation_coeffs_x(),
                    ray_query.get_interpolation_coeffs_y(),
                    _system.get_smooth_convergence(),
                    _system.get_shear());

      qcl::check_cl_error(_ctx->get_command_queue().finish(),
                          "Error while waiting for the lensing query "
                          "to finish.");

      std::cout << "  okay." << std::endl;

    }


    return screen.get_screen();
  }

private:
  qcl::device_context_ptr _ctx;
  lensing_system _system;
  lensing_tree _tree;

};

}

#endif

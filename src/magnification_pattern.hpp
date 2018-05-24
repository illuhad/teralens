
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
#include <fstream>

#include <QCL/qcl_array.hpp>
#include <SpatialCL/query.hpp>
#include <ostream>

#include "configuration.hpp"
#include "lensing_tree.hpp"
#include "lensing_tree_walk.hpp"
#include "brute_force.hpp"
#include "timer.hpp"

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
                 scalar overshooting_factor = 1.f,
                 scalar lens_and_ray_region_size_ratio = 1.5f)
    : _convergence_c{convergence_stars},
      _convergence_s{convergence_smooth},
      _shear{shear},
      _mag_pattern_size{source_plane_size},
      _fixed_overshooting{6.f},
      _overshooting_factor{overshooting_factor},
      _lens_distribution_size_factor{lens_and_ray_region_size_ratio}
  {
    assert(_mag_pattern_size > 0.0f);
    assert(lens_and_ray_region_size_ratio > 1.0f);

    _stars = this->sample_particles(ctx, mean_particle_mass, random_seed);
  }

  lensing_system(const qcl::device_context_ptr& ctx,
                 const std::string& star_list_filename,
                 scalar convergence_smooth,
                 scalar shear,
                 scalar source_plane_size,
                 scalar overshooting_factor = 1.f,
                 scalar lens_and_ray_region_size_ratio = 1.5f)
    : _convergence_s{convergence_smooth},
      _shear{shear},
      _mag_pattern_size{source_plane_size},
      _fixed_overshooting{6.f},
      _overshooting_factor{overshooting_factor},
      _lens_distribution_size_factor{lens_and_ray_region_size_ratio}
  {
    this->load_star_dump(ctx, star_list_filename);
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
    result.s[0] = (_overshooting_factor*_mag_pattern_size + _fixed_overshooting)
        / std::abs(1.0f - gamma - kappa);
    result.s[1] = (_overshooting_factor*_mag_pattern_size + _fixed_overshooting)
        / std::abs(1.0f + gamma - kappa);

    return result;
  }

  const qcl::device_array<particle_type>& get_particles() const
  {
    return _stars;
  }

  const std::vector<particle_type>& get_host_particles() const
  {
    return _host_stars;
  }

  scalar get_source_plane_size() const
  {
    return _mag_pattern_size;
  }

  void write_star_dump(const std::string& filename) const
  {
    std::ofstream dump_file(filename.c_str());
    if(!dump_file.is_open())
      throw std::runtime_error{"Could not open file for writing: "+filename};

    for(const auto& particle : _host_stars)
    {
      dump_file << particle.s[0] << " "
                << particle.s[1] << " "
                << particle.s[2] << std::endl;
    }
  }

private:

  scalar _convergence_c;
  scalar _convergence_s;
  scalar _shear;
  scalar _mag_pattern_size;
  scalar _fixed_overshooting;
  scalar _overshooting_factor;
  scalar _lens_distribution_size_factor;
  qcl::device_array<particle_type> _stars;
  std::vector<particle_type> _host_stars;

  void load_star_dump(const qcl::device_context_ptr& ctx,
                      const std::string& filename)
  {
    std::ifstream dump_file(filename.c_str());
    if(!dump_file.is_open())
      throw std::runtime_error{"Could not open file for reading: "+filename};

    _host_stars.resize(0);

    vector2 center_of_mass{0.0f, 0.0f};
    scalar total_mass = 0.0f;

    while(dump_file.good())
    {
      scalar x,y,mass;
      dump_file >> x;
      dump_file >> y;
      dump_file >> mass;

      if(dump_file.good())
      {
        particle_type p;
        p.s[0] = x;
        p.s[1] = y;
        p.s[2] = mass;

        center_of_mass.s[0] += mass * x;
        center_of_mass.s[1] += mass * y;
        total_mass += mass;

        _host_stars.push_back(p);
      }
    }

    center_of_mass.s[0] /= total_mass;
    center_of_mass.s[1] /= total_mass;

    scalar max_radius = 0.0f;
    for(auto& p : _host_stars)
    {
      for(std::size_t i = 0; i < 2; ++i)
        p.s[i] -= center_of_mass.s[i];
      scalar r2 = p.s[0]*p.s[0]+p.s[1]*p.s[1];
      if(r2 > max_radius)
        max_radius = r2;
    }
    max_radius = std::sqrt(max_radius);

    this->_convergence_c = total_mass / (max_radius * max_radius);
    this->_stars = qcl::device_array<particle_type>{ctx, _host_stars};
  }

  qcl::device_array<particle_type> sample_particles(const qcl::device_context_ptr& ctx,
                                                    scalar mean_particle_mass,
                                                    std::size_t seed)
  {
    vector2 shooting_region = this->get_shooting_region_extent();
    scalar lens_radius =
        _lens_distribution_size_factor * 0.5f *
        std::sqrt(shooting_region.s[0]*shooting_region.s[0]
                + shooting_region.s[1]*shooting_region.s[1]);

    std::size_t num_particles =
        static_cast<std::size_t>(
          std::round(_convergence_c * lens_radius * lens_radius / mean_particle_mass));

    this->_host_stars = std::vector<particle_type>(num_particles);

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

      _host_stars[i] = p;
    }

    return qcl::device_array<particle_type>{ctx, _host_stars};
  }
};

/*
using grouped_dfs_query_engine = spatialcl::query::engine::grouped_depth_first
<
  lensing_tree,
  primary_ray_query
>;*/


using dfs_query_engine = spatialcl::query::engine::depth_first
<
  lensing_tree,
  primary_ray_query,
  spatialcl::query::engine::HIERARCHICAL_ITERATION_RELAXED,
  primary_ray_query_group_size
>;

class magnification_pattern_generator
{
public:

  magnification_pattern_generator(const qcl::device_context_ptr& ctx,
                                  const lensing_system& system,
                                  std::ostream& log)
    : _ctx{ctx},
      _system{system},
      _log{log},
      _runtime{0.0},
      _num_traced_rays{0}
  {
  }

  qcl::device_array<int> run(std::size_t resolution,
                             scalar num_primary_rays_ppx,
                             scalar tree_opening_angle = 0.4,
                             std::size_t brute_force_threshold = max_brute_force_lenses)
  {
    // Calculate lower-left corner of the shooting region
    vector2 shooting_region_min_corner = _system.get_shooting_region_extent();
    for(int i = 0; i < 2; ++i)
      shooting_region_min_corner.s[i] *= -0.5f;

    // Calculate distance between primary rays
    double px_size_x =
        static_cast<double>(_system.get_shooting_region_extent().s[0]) / resolution;
    double px_size_y =
        static_cast<double>(_system.get_shooting_region_extent().s[1]) / resolution;
    scalar ray_distance = std::sqrt(px_size_x * px_size_y / num_primary_rays_ppx);

    bool use_tree = _system.get_particles().size() > brute_force_threshold;

    if(!use_tree)
    {
      std::size_t secondary_rays_per_primary_ray = 2 * secondary_rays_per_cell;
      ray_distance /= secondary_rays_per_primary_ray;
    }

    std::size_t num_rays_x =
        static_cast<std::size_t>(
          std::round(_system.get_shooting_region_extent().s[0] / ray_distance));
    std::size_t num_rays_y =
        static_cast<std::size_t>(
          std::round(_system.get_shooting_region_extent().s[1] / ray_distance));

    // Round the number of rays to a multiple of 8, to make sure
    // that an entire tile is sampled (see the ray position generation code)
    num_rays_x = make_multiple_of(num_rays_x, 8);
    num_rays_y = make_multiple_of(num_rays_y, 8);
    // Correct ray distance

    _num_traced_rays = num_rays_x * num_rays_y;
    if(use_tree)
    {
      _num_traced_rays *=
          num_interpolation_cells * secondary_rays_per_cell * secondary_rays_per_cell;
    }

    // If we are using the brute-force backend, use a larger batch size
    std::size_t effective_max_batch_size = max_batch_size;

    if(!use_tree)
      effective_max_batch_size *= 16;

    timer t;
    t.start();

    // Create a ray_scheduler object, which will be used to calculate
    // the coordinates of the primary rays
    ray_scheduler scheduler{
      _ctx,
      shooting_region_min_corner,
      ray_distance,
      num_rays_x,
      num_rays_y,
      effective_max_batch_size
    };

    // Create pixel screen object, which will count the rays in each pixel
    pixel_screen screen{
      _ctx,
      resolution,
      resolution,
      {{0.0f, 0.0f}},
      {{_system.get_source_plane_size(), _system.get_source_plane_size()}}
    };

    for(int i = 0; !scheduler.all_rays_processed(); ++i)
    {
      _log << "--> Tracing rays of batch " << i << std::endl;

      // Generate a batch of primary ray positions on the GPU
      std::size_t num_rays_in_batch = scheduler.generate_ray_batch();
      _log << "  Scheduled " << num_rays_in_batch << " primary rays." << std::endl;

      if(use_tree)
      {
        this->solve_with_tree(scheduler,
                              num_rays_in_batch,
                              ray_distance,
                              tree_opening_angle,
                              screen);
      }
      else
      {
        this->solve_with_brute_force(scheduler,
                                     num_rays_in_batch,
                                     ray_distance,
                                     screen);
      }

    }


    qcl::check_cl_error(_ctx->get_command_queue().finish(),
                        "Error while waiting for the lensing query "
                        "to finish.");

    _runtime = t.stop();

    return screen.get_screen();
  }

  double get_last_runtime() const
  {
    return _runtime;
  }

  std::size_t get_last_num_traced_rays() const
  {
    return _num_traced_rays;
  }

private:
  std::size_t make_multiple_of(std::size_t x, std::size_t i) const
  {
    if(x % i == 0)
      return x;

    return (x/i + 1) * i;
  }

  void solve_with_tree(const ray_scheduler& scheduler,
                       const std::size_t num_rays_in_batch,
                       const scalar ray_distance,
                       const scalar tree_opening_angle,
                       pixel_screen& screen)
  {

    if(this->_tree == nullptr)
      _tree = std::make_shared<lensing_tree>(_ctx, _system.get_particles());

    dfs_query_engine lensing_query_engine;
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
    _log << "  Running tree queries for primary rays..." << std::endl;
    lensing_query_engine(*_tree, ray_query);

    // Create secondary ray tracer, which will sample interpolation cells
    // for the long-range deflections and evaluate the close-range
    // deflections from close stars at many locations for each primary ray
    secondary_ray_tracer<max_selected_particles> ray_evaluator {
      _ctx,
      &screen
    };

    // Run secondary ray tracer
    _log << "  Tracing secondary rays..." << std::endl;
    ray_evaluator(scheduler.get_current_batch(),
                  num_rays_in_batch,
                  ray_distance,
                  ray_query.get_num_selected_particles(),
                  ray_query.get_selected_particles(),
                  ray_query.get_interpolation_coeffs_x(),
                  ray_query.get_interpolation_coeffs_y(),
                  _system.get_smooth_convergence(),
                  _system.get_shear());

  }

  void solve_with_brute_force(const ray_scheduler& scheduler,
                              const std::size_t num_rays_in_batch,
                              const scalar ray_distance,
                              pixel_screen& screen) const
  {
    brute_force_ray_tracer ray_tracer{
      _ctx,
      _system.get_particles(),
      _system.get_particles().size(),
      &screen
    };

    ray_tracer(scheduler.get_current_batch(),
               num_rays_in_batch,
               _system.get_smooth_convergence(),
               _system.get_shear());

    qcl::check_cl_error(_ctx->get_command_queue().finish(),
                        "Error while waiting for the brute force query "
                        "to finish.");


  }

  qcl::device_context_ptr _ctx;
  lensing_system _system;
  std::shared_ptr<lensing_tree> _tree;

  std::ostream& _log;

  double _runtime;
  std::size_t _num_traced_rays;

};

}

#endif

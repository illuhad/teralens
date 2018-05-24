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

#ifndef BRUTE_FORCE_HPP
#define BRUTE_FORCE_HPP

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>

#include "configuration.hpp"
#include "pixel_screen.hpp"

namespace teralens {

/// Implements a simple brute force strategy for the evaluation of the lens
/// equation by directly summing up the deflection angles.
class brute_force_ray_tracer
{
public:
  QCL_MAKE_MODULE(brute_force_ray_tracer)

  /// Construct object
  /// \param ctx The device context
  /// \param particles The particle device buffer
  /// \param num_particles The number of entries in the particle device
  /// buffer that are actually used
  /// \param screen The pixel screen where the sampled rays will be binned.
  /// Note that it is the user's responsibility to zero out this buffer,
  /// if a new magnification pattern should be calculated.
  brute_force_ray_tracer(const qcl::device_context_ptr& ctx,
                         const qcl::device_array<particle_type>& particles,
                         std::size_t num_particles,
                         pixel_screen* screen)
    : _ctx{ctx},
      _particles{particles},
      _num_particles{num_particles},
      _screen{screen}
  {}

  /// Execute ray tracing
  /// \param ray_positions Contains the positions in the image plane of the rays
  /// \param num_rays The number of rays in the \c ray_positions buffer that
  /// are actually used
  /// \param convergence_smooth The kappa_smooth parameter, i.e. the normalized
  /// surface density due to smoothly distributed matter
  /// \param shear The global shear. It is assumed that the shear compresses
  /// along the x-axis and stretches along the y-axis.
  void operator()(const qcl::device_array<vector2>& ray_positions,
                  std::size_t num_rays,
                  const scalar convergence_smooth,
                  const scalar shear)
  {
    cl_int err = brute_force_lens_equation(_ctx, num_rays, group_size)(
            _particles,
            static_cast<cl_ulong>(_num_particles),
            ray_positions,
            static_cast<cl_ulong>(num_rays),
            convergence_smooth,
            shear,
            _screen->get_screen(),
            static_cast<cl_ulong>(_screen->get_num_pix_x()),
            static_cast<cl_ulong>(_screen->get_num_pix_y()),
            _screen->get_screen_center(),
            _screen->get_screen_extent());

    qcl::check_cl_error(err, "Could not enqueue brute_force_lens_equation kernel");
  }

private:
  static constexpr std::size_t group_size = 128;


  QCL_ENTRYPOINT(brute_force_lens_equation)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(pixel_screen)
    QCL_IMPORT_TYPE(particle_type)
    QCL_IMPORT_TYPE(vector2)
    QCL_IMPORT_TYPE(scalar)
    QCL_IMPORT_CONSTANT(group_size)
    QCL_RAW(
      /// Kernel for the evaluation of the lens equation and ray binning in pixels
      __kernel void brute_force_lens_equation(__global particle_type* restrict particles,
                                              const ulong num_particles,
                                              __global vector2* restrict ray_positions,
                                              const ulong num_rays,
                                              const scalar convergence_smooth,
                                              const scalar shear,
                                              __global int* restrict pixels,
                                              const ulong num_pix_x,
                                              const ulong num_pix_y,
                                              const vector2 screen_center,
                                              const vector2 screen_extent)
      {
        const size_t tid = get_global_id(0);
        const size_t lid = get_local_id(0);

        const vector2 ray_position =
          (tid < num_rays) ? ray_positions[tid] : (vector2)0.0f;

        vector2 deflection = (vector2)0.0f;

        __local particle_type particle_cache [group_size];

        for(int particle_offset = 0;
            particle_offset < num_particles;
            particle_offset += group_size)
        {
          const int particle_id = particle_offset + lid;

          if(particle_id < num_particles)
            particle_cache[lid] = particles[particle_id];
          else
            particle_cache[lid] = (particle_type)0.0f;

          barrier(CLK_LOCAL_MEM_FENCE);

          for(int i = 0; i < group_size; ++i)
          {
            const particle_type p = particle_cache[i];
            const vector2 R = ray_position - p.xy;
            deflection += p.z * native_recip(dot(R,R)) * R;
          }

          barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Calculate impact position
        const vector2 impact_position =
          (vector2)(1.f - shear - convergence_smooth,
                    1.f + shear - convergence_smooth) * ray_position - deflection;

        if(tid < num_rays)
          // Count pixel
          count_ray_impact(impact_position,
                           pixels,
                           num_pix_x, num_pix_y,
                           screen_center, screen_extent);
      }
    )
  )

  qcl::device_context_ptr _ctx;
  qcl::device_array<particle_type> _particles;

  std::size_t _num_particles;

  pixel_screen* _screen;
};

}

#endif

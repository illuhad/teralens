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

#ifndef LENSING_TREE_WALK
#define LENSING_TREE_WALK

#include <cassert>

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>
#include <QCL/qcl_array.hpp>

#include <SpatialCL/configuration.hpp>
#include <SpatialCL/query/query_base.hpp>

#include "configuration.hpp"
#include "lensing_moments.hpp"

namespace teralens {

class ray_grid_query : public spatialcl::query::basic_query
{
public:
  QCL_MAKE_MODULE(ray_grid_query)


  ray_grid_query(const qcl::device_context_ptr& ctx,
                 const vector2& shooting_region_center,
                 const vector2& shooting_region_extent,
                 const vector2& primary_ray_separation,
                 const vector2& pixel_screen_center,
                 scalar pixel_size,
                 std::size_t num_px_x,
                 std::size_t num_px_y,
                 scalar shear,
                 scalar convergence,
                 scalar opening_angle)
      : _shooting_region_center{shooting_region_center},
        _shooting_region_extent{shooting_region_extent},
        _primary_ray_separation{primary_ray_separation},

        _shear{shear},
        _convergence{convergence},
        _opening_angle_squared{opening_angle * opening_angle},

        _num_px_x{num_px_x},
        _num_px_y{num_px_y},

        _pixel_screen{ctx, num_px_x * num_px_y},
        _pixel_size{pixel_size},
        _pixel_screen_center{pixel_screen_center}
  {
    assert(primary_ray_separation.s[0] > 0.0f &&
           primary_ray_separation.s[1] > 0.0f);
    _num_rays_x = shooting_region_extent.s[0]/primary_ray_separation.s[0];
    _num_rays_y = shooting_region_extent.s[1]/primary_ray_separation.s[1];

    assert(_num_rays_x > 0);
    assert(_num_rays_y > 0);
    assert(_num_px_x > 0);
    assert(_num_px_y > 0);

    init_pixel_screen(ctx,
                      _num_px_x*_num_px_y,
                      256)(
                this->_pixel_screen,
                static_cast<cl_ulong>(_num_px_x),
                static_cast<cl_ulong>(_num_px_y));
  }

  virtual void push_full_arguments(qcl::kernel_call& call) override
  {
    vector_type shooting_region_min_corner = this->_shooting_region_center;
    vector_type screen_min_corner          = this->_pixel_screen_center;
    for(std::size_t i = 0; i < 2; ++i)
      shooting_region_min_corner.s[i] -= 0.5f * this->_shooting_region_extent.s[i];

    screen_min_corner.s[0] -= 0.5f * this->_pixel_size * _num_px_x;
    screen_min_corner.s[1] -= 0.5f * this->_pixel_size * _num_px_y;

    call.partial_argument_list(shooting_region_min_corner,
                               this->_primary_ray_separation,
                               static_cast<cl_ulong>(_num_rays_x),
                               static_cast<cl_ulong>(_num_rays_y),
                               _shear,
                               _convergence,
                               _opening_angle_squared,
                               _pixel_screen,
                               static_cast<cl_uint>(_num_px_x),
                               static_cast<cl_uint>(_num_px_y),
                               _pixel_size,
                               screen_min_corner);
  }

  virtual std::size_t get_num_independent_queries() const override
  {
    return this->get_total_num_rays();
  }

  virtual ~ray_grid_query(){}

  std::size_t get_num_rays_x() const
  {
    return _num_rays_x;
  }

  std::size_t get_num_rays_y() const
  {
    return _num_rays_y;
  }

  std::size_t get_total_num_rays() const
  {
    return _num_rays_x * _num_rays_y;
  }

  qcl::device_array<cl_int>& get_pixel_screen()
  {
    return _pixel_screen;
  }

  const qcl::device_array<cl_int>& get_pixel_screen() const
  {
    return _pixel_screen;
  }
private:


  vector2 _shooting_region_center;
  vector2 _shooting_region_extent;
  vector2 _primary_ray_separation;

  std::size_t _num_rays_x;
  std::size_t _num_rays_y;

  scalar _shear;
  scalar _convergence;
  scalar _opening_angle_squared;

  std::size_t _num_px_x;
  std::size_t _num_px_y;
  qcl::device_array<cl_int> _pixel_screen;
  scalar _pixel_size;
  vector2 _pixel_screen_center;

  static constexpr std::size_t tile_size = 8;

  QCL_ENTRYPOINT(init_pixel_screen)
  QCL_MAKE_SOURCE
  (
    QCL_IMPORT_CONSTANT(tile_size)
    QCL_INCLUDE_MODULE(spatialcl::configuration<type_system>)
    QCL_INCLUDE_MODULE(lensing_moments)
    QCL_RAW(
      __kernel void init_pixel_screen(__global int* screen,
                                      ulong num_px_x,
                                      ulong num_px_y)
      {
        size_t tid = get_global_id(0);
        if(tid < num_px_x * num_px_y)
          screen[tid] = 0;
      }
    )
    QCL_PREPROCESSOR(define,
      dfs_node_selector(selection_result_ptr,
                        current_node_key_ptr,
                        node_index,
                        node0,
                        node1)
      {
        const vector_type R = CENTER_OF_MASS(node0)-ray_position;

        *selection_result_ptr =
          NODE_WIDTH(node0)*NODE_WIDTH(node0) > opening_angle_squared*dot(R,R);
      }
    )
    QCL_PREPROCESSOR(define,
      dfs_particle_processor(selection_result_ptr,
                             particle_idx,
                             current_particle)
      {
        *selection_result_ptr = 0;

        const vector_type R = ray_position - current_particle.xy;

        deflection += current_particle.z * R * native_recip(dot(R,R));
      }
    )
    QCL_PREPROCESSOR(define,
      dfs_unique_node_discard_event(node_idx,
                                    node0,
                                    node1)
      {
        // Calculate multipoles
        lensing_multipole_expansion expansion;
        EXPANSION_LO(expansion) = node0;
        EXPANSION_HI(expansion) = node1;

        // ToDo verify sign
        deflection -= multipole_expansion_evaluate(expansion, ray_position);
        //vector_type R = ray_position - CENTER_OF_MASS(expansion);
        //deflection += (MASS(expansion) * native_recip(dot(R,R))) * R;
      }
    )
    R"(
      #define declare_full_query_parameter_set() \
        const vector_type shooting_region_min_corner,\
        const vector_type ray_separation,        \
        const ulong num_rays_x,                  \
        const ulong num_rays_y,                  \
        const scalar shear,                      \
        const scalar convergence,                \
        const scalar opening_angle_squared,      \
        __global int* ray_count_pixels,          \
        const uint num_screen_px_x,              \
        const uint num_screen_px_y,              \
        const scalar pixel_size,                 \
        const vector_type screen_min_corner
    )"
    QCL_PREPROCESSOR(define,
      at_query_init()
        vector_type deflection = (vector_type)0.0f;
        const uint ray_id_x = get_query_id() % num_rays_x;
        const uint ray_id_y = get_query_id() / num_rays_x;
        // ToDo: Apply tiling optimization
        const vector_type ray_position = (vector_type)(
                      shooting_region_min_corner.x + ray_id_x * ray_separation.x,
                      shooting_region_min_corner.y + ray_id_y * ray_separation.y);
    )
    QCL_PREPROCESSOR(define,
      at_query_exit()
        if(get_query_id() < num_rays_x * num_rays_y)
        {
          vector_type source_plane_position =
                         (vector_type)(1-shear-convergence,
                                       1+shear-convergence)*ray_position - deflection;
          int2 pixel = convert_int2((source_plane_position - screen_min_corner)/pixel_size);


          if(pixel.x >= 0 &&
             pixel.y >= 0 &&
             pixel.x < num_screen_px_x &&
             pixel.y < num_screen_px_y)
            atomic_inc(ray_count_pixels + pixel.y * num_screen_px_x + pixel.x);
        }
    )
    QCL_PREPROCESSOR(define,
      get_num_queries()
        (num_rays_x * num_rays_y)
    )
  )
};

}

#endif

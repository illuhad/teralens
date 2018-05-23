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
#include "interpolation.hpp"
#include "pixel_screen.hpp"

namespace teralens {

class primary_ray_query : public spatialcl::query::basic_query
{
public:
  QCL_MAKE_MODULE(primary_ray_query)

  primary_ray_query(const qcl::device_context_ptr& ctx,
                    const qcl::device_array<vector_type>& ray_coordinates,
                    std::size_t num_rays,
                    std::size_t max_exact_particles,
                    scalar ray_separation,
                    scalar tree_opening_angle)
      : _ctx{ctx},
        _ray_coords{ray_coordinates},
        _num_rays{num_rays},
        _max_exact_particles{max_exact_particles},
        _exact_particles    {ctx, num_rays * max_exact_particles},
        _num_selected_particles{ctx, num_rays},
        _opening_angle {tree_opening_angle},
        _coeffs_x{ctx, num_rays * 4},
        _coeffs_y{ctx, num_rays * 4},
        _ray_separation{ray_separation}
  {}

  virtual void push_full_arguments(qcl::kernel_call& call) override
  {
    call.partial_argument_list(_ray_coords,
                               static_cast<cl_ulong>(_num_rays),
                               _exact_particles,
                               _num_selected_particles,
                               _coeffs_x,
                               _coeffs_y,
                               static_cast<cl_uint>(_max_exact_particles),
                               _ray_separation,
                               _opening_angle * _opening_angle);
  }

  virtual std::size_t get_num_independent_queries() const override
  {
    return _num_rays;
  }

  virtual ~primary_ray_query(){}

  const qcl::device_array<particle_type>& get_selected_particles() const
  {
    return _exact_particles;
  }

  const qcl::device_array<bicubic_interpolation_coefficients>&
  get_interpolation_coeffs_x() const
  {
    return _coeffs_x;
  }

  const qcl::device_array<bicubic_interpolation_coefficients>&
  get_interpolation_coeffs_y() const
  {
    return _coeffs_y;
  }

  const qcl::device_array<cl_uint>& get_num_selected_particles() const
  {
    return _num_selected_particles;
  }


private:
  qcl::device_context_ptr _ctx;
  qcl::device_array<vector_type> _ray_coords;
  const std::size_t _num_rays;
  const std::size_t _max_exact_particles;

  qcl::device_array<particle_type> _exact_particles;
  qcl::device_array<cl_uint> _num_selected_particles;

  scalar _opening_angle;

  qcl::device_array<bicubic_interpolation_coefficients> _coeffs_x;
  qcl::device_array<bicubic_interpolation_coefficients> _coeffs_y;

  scalar _ray_separation;

  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(spatialcl::configuration<type_system>)
    QCL_INCLUDE_MODULE(lensing_moments)
    QCL_INCLUDE_MODULE(interpolation)
    QCL_IMPORT_TYPE(vector2)
    QCL_IMPORT_TYPE(vector8)
    R"(
      #define NUM_INTERPOLATION_POINTS 5
      #define HALF_NUM_INTERPOLATION_POINTS 2

      #define declare_full_query_parameter_set()            \
        __global vector_type* restrict ray_positions,       \
        const ulong num_rays,                               \
        __global particle_type* restrict selected_particles,\
        __global uint* restrict selected_particles_counter, \
        __global bicubic_interpolation_coefficients* restrict coeffs_x,\
        __global bicubic_interpolation_coefficients* restrict coeffs_y,\
        const uint max_selected_particles,                  \
        const scalar ray_separation,                        \
        const scalar opening_angle_squared
    )"
    QCL_PREPROCESSOR(define,
      get_num_queries()
        (num_rays)
    )
    QCL_PREPROCESSOR(define,
      at_query_init()
        vector_type ray_position = (vector_type)0.0f;

        if(get_query_id() < num_rays)
          ray_position = ray_positions[get_query_id()];

        uint num_selected_particles = 0;

        scalar long_range_deflections_x [NUM_INTERPOLATION_POINTS][NUM_INTERPOLATION_POINTS];
        scalar long_range_deflections_y [NUM_INTERPOLATION_POINTS][NUM_INTERPOLATION_POINTS];
        for(int i = 0; i < NUM_INTERPOLATION_POINTS; ++i)
          for(int j = 0; j < NUM_INTERPOLATION_POINTS; ++j)
          {
            long_range_deflections_x[i][j] = 0.0f;
            long_range_deflections_y[i][j] = 0.0f;
          }
    )
    QCL_PREPROCESSOR(define,
      dfs_node_selector(selection_result_ptr,
                        current_node_key_ptr,
                        node_index,
                        node0,
                        node1)
      {
        // The NODE_EXTENT field is overwritten after the tree construction
        // with the geometric center of the node
        const vector_type R = NODE_EXTENT(node0) - ray_position;

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

        // Save particle
        // ToDo: Make sure tid is valid? This would enable the use of
        // the grouped_dfs_engine
        selected_particles[get_query_id() * max_selected_particles + num_selected_particles] =
                         current_particle;

        num_selected_particles = min(num_selected_particles + 1,
                                     max_selected_particles - 1);
      }
    )
    QCL_PREPROCESSOR(define,
      dfs_unique_node_discard_event(node_idx,
                                    node0,
                                    node1)
      {
        // Evaluate at interpolation points
        lensing_multipole_expansion expansion;
        EXPANSION_LO(expansion) = node0;
        EXPANSION_HI(expansion) = node1;
        for(int i = 0; i < NUM_INTERPOLATION_POINTS; ++i)
        {
          for(int j = 0; j < NUM_INTERPOLATION_POINTS; ++j)
          {
            vector2 evaluated_position = ray_position;
            evaluated_position.x += (i-HALF_NUM_INTERPOLATION_POINTS)*0.5f*ray_separation;
            evaluated_position.y += (j-HALF_NUM_INTERPOLATION_POINTS)*0.5f*ray_separation;

            // Store deflection in evaluated position to save registers
            evaluated_position = multipole_expansion_evaluate(expansion,
                                                              evaluated_position);
            long_range_deflections_x[i][j] -= evaluated_position.x;
            long_range_deflections_y[i][j] -= evaluated_position.y;
          }
        }
      }
    )
    QCL_PREPROCESSOR(define,
      at_query_exit()
      {
        selected_particles_counter[get_query_id()] = num_selected_particles;

        // Calculate interpolation coefficients
        scalar deflections [4][4];
        for(int offset_y = 0; offset_y < 2; ++offset_y)
        {
          for(int offset_x = 0; offset_x < 2; ++offset_x)
          {
            bicubic_interpolation_coefficients coeffs;
            for(int i = 0; i < 4; ++i)                           
              for(int j = 0; j < 4; ++j)
                deflections[i][j] = long_range_deflections_x[offset_x + i][offset_y + j];

            coeffs = bicubic_interpolation_init(*(vector16*)deflections);
            coeffs_x[4*get_query_id() + 2*offset_y + offset_x] = coeffs;

            for(int i = 0; i < 4; ++i)
              for(int j = 0; j < 4; ++j)
                deflections[i][j] = long_range_deflections_y[offset_x + i][offset_y + j];

            coeffs = bicubic_interpolation_init(*(vector16*)deflections);
            coeffs_y[4*get_query_id() + 2*offset_y + offset_x] = coeffs;
          }
        }
      }
    )
  )
};


// Max_particles_per_ray must be a power of 2.
template<std::size_t Max_particles_per_ray>
class  secondary_ray_tracer
{
public:
  QCL_MAKE_MODULE(secondary_ray_tracer)

  secondary_ray_tracer(const qcl::device_context_ptr& ctx,
                       pixel_screen* screen)
      : _ctx{ctx},
        _screen{screen}
  {}

  void operator()(const qcl::device_array<vector2>& primary_ray_positions,
                  const std::size_t num_primary_rays,
                  const scalar primary_ray_separation,
                  const qcl::device_array<cl_uint>& num_selected_particles,
                  const qcl::device_array<particle_type>& selected_particles,
                  const qcl::device_array<bicubic_interpolation_coefficients>& coeffs_x,
                  const qcl::device_array<bicubic_interpolation_coefficients>& coeffs_y,
                  const scalar convergence_smooth,
                  const scalar shear)
  {
    assert(num_selected_particles.size() >= num_primary_rays);
    assert(selected_particles.size() >= num_primary_rays * Max_particles_per_ray);


    std::size_t total_num_rays_per_cell = secondary_rays_per_cell*secondary_rays_per_cell;

    cl_int err = this->evaluate_lens_equation(_ctx,
                        cl::NDRange{total_num_rays_per_cell * cells_per_ray * num_primary_rays},
                        cl::NDRange{group_size})(
          primary_ray_positions,
          selected_particles,
          num_selected_particles,
          coeffs_x,
          coeffs_y,
          static_cast<cl_ulong>(num_primary_rays),
          convergence_smooth,
          shear,
          primary_ray_separation,
          _screen->get_screen(),
          static_cast<cl_ulong>(_screen->get_num_pix_x()),
          static_cast<cl_ulong>(_screen->get_num_pix_y()),
          _screen->get_screen_center(),
          _screen->get_screen_extent());

    qcl::check_cl_error(err, "Could not enqueue evaluate_lens_equation kernel.");

  }

private:

  qcl::device_context_ptr _ctx;
  pixel_screen* _screen;

  static constexpr std::size_t group_size = 128;
  static constexpr std::size_t groups_per_cell =
      (secondary_rays_per_cell*secondary_rays_per_cell)/group_size;


  static_assert(group_size >= Max_particles_per_ray, "At least as many threads as the "
                                                     "maximum particle number are required "
                                                     "per primary ray");
  // Number of interpolation cells per primary ray in each dimension
  static constexpr std::size_t num_interpolation_cells = 2;

  static constexpr std::size_t cells_per_ray = num_interpolation_cells
                                             * num_interpolation_cells;

  QCL_ENTRYPOINT(evaluate_lens_equation)
  QCL_MAKE_SOURCE(
    QCL_IMPORT_CONSTANT(Max_particles_per_ray)
    QCL_INCLUDE_MODULE(spatialcl::configuration<type_system>)
    QCL_INCLUDE_MODULE(interpolation)
    QCL_INCLUDE_MODULE(pixel_screen)
    QCL_INCLUDE_MODULE(lensing_moments)
    QCL_IMPORT_TYPE(vector2)
    QCL_IMPORT_TYPE(vector8)
    QCL_IMPORT_CONSTANT(secondary_rays_per_cell)
    QCL_IMPORT_CONSTANT(cells_per_ray)
    QCL_IMPORT_CONSTANT(group_size)
    QCL_IMPORT_CONSTANT(groups_per_cell)
    QCL_RAW(


      // id should be <= 3. Translates id into
      // 2d indices ranging from -1 to 0 in each component
      char2 interpolation_cell_id2d(const uchar id)
      {
        return (char2)((id &  1) - 1,
                       (id >> 1) - 1);
      }

      // Translates the one dimensional id of a ray into
      // a two dimensional id within the interpolation cell,
      // from 0 to secondary_rays_per_cell-1
      uchar2 ray_id2d(const uint ray_id)
      {
        return (uchar2)(ray_id & (secondary_rays_per_cell - 1),
                        ray_id /  secondary_rays_per_cell);
      }

      // Blocksize should be secondary_rays_per_cell^2. Each block
      // will process the secondary rays of one interpolation cell.
      __kernel void evaluate_lens_equation(__global vector2* restrict primary_ray_positions,
                                           __global particle_type* restrict exact_particles,
                                           __global uint* restrict num_exact_particles,
                                           __global bicubic_interpolation_coefficients* restrict coefficients_x,
                                           __global bicubic_interpolation_coefficients* restrict coefficients_y,
                                           const ulong num_primary_rays,
                                           const scalar convergence_smooth,
                                           const scalar shear,
                                           const scalar primary_ray_separation,
                                           __global int* restrict pixels,
                                           const ulong num_px_x,
                                           const ulong num_px_y,
                                           const vector2 screen_center,
                                           const vector2 screen_extent)
      {
        const size_t tid = get_global_id(0);
        const size_t lid = get_local_id(0);
        const size_t gid = get_group_id(0);

        const size_t primary_ray_id       =  gid / (groups_per_cell * cells_per_ray);
        const uchar interpolation_cell_id = (gid /  groups_per_cell) & 3;
        const scalar interpolation_cell_size = primary_ray_separation * 0.5f;
        const scalar secondary_ray_separation = interpolation_cell_size / secondary_rays_per_cell;

        __local particle_type particle_cache [Max_particles_per_ray];

        const vector2 interpolation_cell_min_corner = primary_ray_positions[primary_ray_id]
                   + interpolation_cell_size * convert_float2(interpolation_cell_id2d(interpolation_cell_id));
        const uint num_particles = num_exact_particles[primary_ray_id];

        vector2 deflection = (vector2)0.0f;

        const uchar2 rid = ray_id2d(lid + (gid % groups_per_cell) * group_size);
        const vector2 evaluation_position = interpolation_cell_min_corner
            + secondary_ray_separation*(convert_float2(rid)+(vector2)0.5f);

        // Collectively load exact particles into local memory
        if(lid < num_particles)
          particle_cache[lid] = exact_particles[primary_ray_id * Max_particles_per_ray + lid];

        barrier(CLK_LOCAL_MEM_FENCE);

        // First, calculate close-range deflections
//$pp pragma unroll 4$
        for(int i = 0; i < num_particles; ++i)
        {
          const particle_type p = particle_cache[i];
          const vector2 R = evaluation_position - p.xy;
          deflection += R * native_divide(p.z, dot(R,R));
        }

        // Calculate long-range, interpolated deflections
        {
          const bicubic_interpolation_coefficients coeffs_x =
              coefficients_x[cells_per_ray * primary_ray_id + interpolation_cell_id];
          const bicubic_interpolation_coefficients coeffs_y =
              coefficients_y[cells_per_ray * primary_ray_id + interpolation_cell_id];

          const vector2 relative_position =
              (evaluation_position - interpolation_cell_min_corner) / (vector2)interpolation_cell_size;

          deflection.x += bicubic_unit_square_interpolation(coeffs_x, relative_position);
          deflection.y += bicubic_unit_square_interpolation(coeffs_y, relative_position);
        }

        // Calculate impact position in the source plane
        if(lid < secondary_rays_per_cell*secondary_rays_per_cell)
        {
          const vector2 shear_convergence_term =
                     (vector2)(1.f - shear - convergence_smooth,
                               1.f + shear - convergence_smooth) * evaluation_position;

          // and count the impact in the pixel screen
          count_ray_impact(shear_convergence_term - deflection,
                           pixels,
                           num_px_x, num_px_y,
                           screen_center, screen_extent);


        }
      }

    )
  )

};

class ray_scheduler
{
public:
  QCL_MAKE_MODULE(ray_scheduler)

  ray_scheduler(const qcl::device_context_ptr& ctx,
                const vector2& shooting_region_min_corner,
                const scalar ray_separation,
                const std::size_t num_rays_x,
                const std::size_t num_rays_y,
                const std::size_t batch_size)
    : _ctx{ctx},
      _region_min_corner{shooting_region_min_corner},
      _ray_separation{ray_separation},
      _num_rays_x{num_rays_x},
      _num_rays_y{num_rays_y},
      _num_processed_rays{0},
      _batch_size{batch_size},
      _batch{ctx, batch_size}
  {}

  std::size_t get_num_processed_rays() const
  {
    return _num_processed_rays;
  }

  std::size_t get_num_scheduled_rays() const
  {
    return _num_rays_x * _num_rays_y;
  }

  bool all_rays_processed() const
  {
    return get_num_processed_rays() >= get_num_scheduled_rays();
  }

  std::size_t generate_ray_batch()
  {
    std::size_t num_rays = std::min(_batch_size,
                                    get_num_scheduled_rays() - get_num_processed_rays());

    cl_int err = this->generate_ray_positions(_ctx,
                                              cl::NDRange{num_rays},
                                              cl::NDRange{256})(
          _batch,
          static_cast<cl_ulong>(num_rays),
          static_cast<cl_ulong>(_num_processed_rays),
          static_cast<cl_ulong>(_num_rays_x),
          static_cast<cl_ulong>(_num_rays_y),
          _ray_separation,
          _region_min_corner);

    qcl::check_cl_error(err, "Could not enqueue generate_ray_positions kernel");

    _num_processed_rays += num_rays;

    return num_rays;
  }

  const qcl::device_array<vector2>& get_current_batch() const
  {
    return _batch;
  }

private:

  QCL_ENTRYPOINT(generate_ray_positions)
  QCL_MAKE_SOURCE(
    QCL_IMPORT_CONSTANT(reorder_primary_rays)
    QCL_IMPORT_TYPE(vector2)
    QCL_IMPORT_TYPE(scalar)
    QCL_RAW(
      __kernel void generate_ray_positions(__global vector2* restrict out,
                                           const ulong num_rays,
                                           const ulong start_ray,
                                           const ulong num_rays_x,
                                           const ulong num_rays_y,
                                           const scalar ray_separation,
                                           const vector2 screen_min_corner)
      {
        const ulong tid = get_global_id(0);

        if(tid < num_rays)
        {
          const ulong ray_id = tid + start_ray;

$pp if reorder_primary_rays == 1 $
          const uint num_tiles_x = num_rays_x >> 3;

          // First calculate the index of the 8x8 tile to which
          // this ray belongs
          const ulong tile_id  = ray_id >> 6;
          const uint tile_id_x = tile_id % num_tiles_x;
          const uint tile_id_y = tile_id / num_tiles_x;

          const uint local_ray_id = ray_id & 63;

          // Then calculate the ray position by sorting
          // along a z-curve within each tile
          const uint rid_x =  (tile_id_x << 3)
                           |  (local_ray_id & 1)
                           | ((local_ray_id & 4) >> 1)
                           | ((local_ray_id & 16)>> 2);
          const uint rid_y =  (tile_id_y << 3)
                           | ((local_ray_id & 2) >> 1)
                           | ((local_ray_id & 8) >> 2)
                           | ((local_ray_id & 32)>> 3);
          //const uint rid_x = 8 * tile_id_x + local_ray_id % 8;
          //const uint rid_y = 8 * tile_id_y + local_ray_id / 8;
$pp else $
          const uint rid_x = ray_id % num_rays_x;
          const uint rid_y = ray_id / num_rays_x;
$pp endif $

          const vector2 pos = screen_min_corner
                   + (vector2)(rid_x, rid_y) * (vector2)(ray_separation, ray_separation);

          out[tid] = pos;
        }
      }
    )
  )

  qcl::device_context_ptr _ctx;

  const vector2 _region_min_corner;

  const scalar _ray_separation;
  const std::size_t _num_rays_x;
  const std::size_t _num_rays_y;

  std::size_t _num_processed_rays;

  const std::size_t _batch_size;

  qcl::device_array<vector2> _batch;
};

}

#endif

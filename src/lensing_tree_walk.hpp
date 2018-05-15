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

namespace teralens {

class primary_ray_query : public spatialcl::query::basic_query
{
public:
  QCL_MAKE_MODULE(primary_ray_query)

  primary_ray_query(const qcl::device_context_ptr& ctx,
                    const qcl::device_array<vector_type>& ray_coordinates,
                    std::size_t num_rays,
                    std::size_t max_exact_particles,
                    std::size_t max_nodes,
                    scalar tree_opening_angle)
      : _ctx{ctx},
        _ray_coords{ray_coordinates},
        _num_rays{num_rays},
        _max_exact_particles{max_exact_particles},
        _max_nodes{max_nodes},
        _exact_particles    {ctx, num_rays * max_exact_particles},
        _approximated_nodes0{ctx, num_rays * max_nodes},
        _approximated_nodes1{ctx, num_rays * max_nodes},
        _num_selected_particles{ctx, num_rays},
        _num_selected_nodes    {ctx, num_rays},
        _opening_angle {tree_opening_angle}
  {}

  virtual void push_full_arguments(qcl::kernel_call& call) override
  {
    call.partial_argument_list(_ray_coords,
                               static_cast<cl_ulong>(_num_rays),
                               _exact_particles,
                               _approximated_nodes0,
                               _approximated_nodes1,
                               _num_selected_particles,
                               _num_selected_nodes,
                               static_cast<cl_uint>(_max_exact_particles),
                               static_cast<cl_uint>(_max_nodes),
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

  const qcl::device_array<vector8>& get_selected_nodes0() const
  {
    return _approximated_nodes0;
  }

  const qcl::device_array<vector8>& get_selected_nodes1() const
  {
    return _approximated_nodes1;
  }

  const qcl::device_array<cl_uint>& get_num_selected_particles() const
  {
    return _num_selected_particles;
  }

  const qcl::device_array<cl_uint>& get_num_selected_nodes() const
  {
    return _num_selected_nodes;
  }

private:
  qcl::device_context_ptr _ctx;
  qcl::device_array<vector_type> _ray_coords;
  const std::size_t _num_rays;
  const std::size_t _max_exact_particles;
  const std::size_t _max_nodes;

  qcl::device_array<particle_type> _exact_particles;
  qcl::device_array<vector8> _approximated_nodes0;
  qcl::device_array<vector8> _approximated_nodes1;
  qcl::device_array<cl_uint> _num_selected_particles;
  qcl::device_array<cl_uint> _num_selected_nodes;

  scalar _opening_angle;

  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(spatialcl::configuration<type_system>)
    QCL_INCLUDE_MODULE(lensing_moments)
    QCL_INCLUDE_MODULE(interpolation)
    QCL_IMPORT_TYPE(vector8)
    R"(
      #define NUM_INTERPOLATION_POINTS

      #define declare_full_query_parameter_set()            \
        __global vector_type* restrict ray_positions,       \
        const ulong num_rays,                               \
        __global particle_type* restrict selected_particles,\
        __global vector8* restrict selected_nodes0,         \
        __global vector8* restrict selected_nodes1,         \
        __global uint* restrict selected_particles_counter, \
        __global uint* restrict selected_nodes_counter,     \
        const uint max_selected_particles,                  \
        const uint max_selected_nodes,                      \
        scalar opening_angle_squared
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
        uint num_selected_nodes     = 0;

        //vector_type long_range_deflections [NUM_INTERPOLATION_POINTS][NUM_INTERPOLATION_POINTS];
    )
    QCL_PREPROCESSOR(define,
      dfs_node_selector(selection_result_ptr,
                        current_node_key_ptr,
                        node_index,
                        node0,
                        node1)
      {
        const vector_type R = CENTER_OF_MASS(node0) - ray_position;

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
        // Save nodes
        // ToDo: Make sure tid is valid? This would enable the use of
        // the grouped_dfs_engine
        ulong i = get_query_id() * max_selected_nodes + num_selected_nodes;

        selected_nodes0[i] = node0;
        selected_nodes1[i] = node1;

        num_selected_nodes = min(num_selected_nodes + 1,
                                 max_selected_nodes - 1);
      }
    )
    QCL_PREPROCESSOR(define,
      at_query_exit()
      {
        selected_particles_counter[get_query_id()] = num_selected_particles;
        selected_nodes_counter    [get_query_id()] = num_selected_nodes;
      }
    )
  )
};

class pixel_screen
{
public:
  QCL_MAKE_MODULE(pixel_screen)

  pixel_screen(const qcl::device_context_ptr& ctx,
               const std::size_t num_pix_x,
               const std::size_t num_pix_y,
               const vector2& pixel_screen_center,
               const vector2& pixel_screen_extent)
    : _ctx{ctx},
      _num_pix_x{num_pix_x},
      _num_pix_y{num_pix_y},
      _pixel_screen{ctx, num_pix_x * num_pix_y},
      _center{pixel_screen_center},
      _extent{pixel_screen_extent}
  {
    cl_int err = this->init_pixel_screen(ctx,
                                         cl::NDRange{num_pix_x*num_pix_y},
                                         cl::NDRange{256})(
          _pixel_screen,
          static_cast<cl_ulong>(_num_pix_x),
          static_cast<cl_ulong>(_num_pix_y));

    qcl::check_cl_error(err, "Could not enqueue init_pixel_screen kernel");
  }

  const qcl::device_array<cl_int>& get_screen() const
  {
    return _pixel_screen;
  }

  std::size_t get_num_pix_x() const
  { return _num_pix_x; }


  std::size_t get_num_pix_y() const
  { return _num_pix_y; }

  vector2 get_screen_center() const
  {
    return _center;
  }

  vector2 get_screen_extent() const
  {
    return _extent;
  }
private:
  qcl::device_context_ptr _ctx;

  std::size_t _num_pix_x;
  std::size_t _num_pix_y;
  vector2 _center;
  vector2 _extent;

  qcl::device_array<cl_int> _pixel_screen;

  QCL_ENTRYPOINT(init_pixel_screen)
  QCL_ENTRYPOINT(count_ray_impacts)
  QCL_MAKE_SOURCE(
    QCL_IMPORT_TYPE(vector2)
    QCL_RAW(
      __kernel void init_pixel_screen(__global int* screen,
                                      ulong num_px_x,
                                      ulong num_px_y)
      {
        size_t tid = get_global_id(0);
        if(tid < num_px_x * num_px_y)
          screen[tid] = 0;
      }



      void count_ray_impact(vector2 pos,
                            __global int* restrict pixels,
                            ulong num_px_x,
                            ulong num_px_y,
                            vector2 screen_center,
                            vector2 screen_extent)
      {

        vector2 screen_min_corner = screen_center - 0.5f * screen_extent;
        vector2 pixel_size = (vector2)(screen_extent.x / num_px_x,
                                       screen_extent.y / num_px_y);

        int2 pixel = convert_int2((pos - screen_min_corner)/pixel_size);

        if(pixel.x >= 0 &&
           pixel.y >= 0 &&
           pixel.x < num_px_x &&
           pixel.y < num_px_y)
          atomic_inc(pixels + pixel.y * num_px_x + pixel.x);
      }

    )
  )
};

// Max_particles_per_ray and Max_nodes_per_ray
// must be a power of 2.
template<std::size_t Max_particles_per_ray,
         std::size_t Max_nodes_per_ray>
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
                  const qcl::device_array<cl_uint>& num_selected_nodes,
                  const qcl::device_array<vector8>& selected_nodes0,
                  const qcl::device_array<vector8>& selected_nodes1,
                  const scalar convergence_smooth,
                  const scalar shear)
  {
    assert(num_selected_particles.size() >= num_primary_rays);
    assert(selected_particles.size() >= num_primary_rays * Max_particles_per_ray);
    assert(selected_nodes0.size() >= num_primary_rays * Max_nodes_per_ray);
    assert(selected_nodes1.size() >= num_primary_rays * Max_nodes_per_ray);
    assert(num_selected_nodes.size() >= num_primary_rays);

    qcl::device_array<vector2> interpolation_deflections{
      _ctx,
      num_primary_rays * num_multipole_evaluations
    };

    cl_int err = this->compute_interpolation_deflections(
          _ctx,
          cl::NDRange{num_primary_rays * multipole_evaluations_group_size},
          cl::NDRange{multipole_evaluations_group_size})(
            selected_nodes0,
            selected_nodes1,
            num_selected_nodes,
            primary_ray_positions,
            static_cast<cl_ulong>(num_primary_rays),
            primary_ray_separation,
            interpolation_deflections);

    qcl::check_cl_error(err, "Could not enqueue compute_interpolation_deflections kernel.");

    qcl::device_array<bicubic_interpolation_coefficients> coeffs_x{
      _ctx,
      cells_per_ray * num_primary_rays
    };

    qcl::device_array<bicubic_interpolation_coefficients> coeffs_y{
      _ctx,
      cells_per_ray * num_primary_rays
    };

    err = this->compute_interpolation_coefficients(_ctx,
                                                   cl::NDRange{cells_per_ray * num_primary_rays},
                                                   cl::NDRange{128})(
          interpolation_deflections,
          static_cast<cl_ulong>(num_primary_rays),
          coeffs_x,
          coeffs_y);
    qcl::check_cl_error(err, "Could not enqueue compute_interpolation_coefficients kernel.");

    std::size_t total_num_rays_per_cell = secondary_rays_per_cell*secondary_rays_per_cell;
    err = this->evaluate_lens_equation(_ctx,
                                       cl::NDRange{total_num_rays_per_cell * cells_per_ray * num_primary_rays},
                                       cl::NDRange{std::max(Max_particles_per_ray,total_num_rays_per_cell)})(
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

  // Number of interpolation cells per primary ray in each dimension
  static constexpr std::size_t num_interpolation_cells = 2;
  // Number of rays that need to be interpolated for the interpolation
  // cells in each direction per primary ray. This is not simply
  // 4*num_interpolation_cells because the boundary evaluations between
  // the cells are shared.
  static constexpr std::size_t interpolation_ray_grid_size=5;

  static constexpr std::size_t num_multipole_evaluations=
      interpolation_ray_grid_size*interpolation_ray_grid_size;
  static constexpr std::size_t multipole_evaluations_group_size = 32;
  static_assert(multipole_evaluations_group_size >= num_multipole_evaluations,
                "The group size must be at least as large as the number"
                " of interpolation samples around a primary ray.");

  static constexpr std::size_t cells_per_ray = num_interpolation_cells
                                             * num_interpolation_cells;

  QCL_ENTRYPOINT(compute_interpolation_deflections)
  QCL_ENTRYPOINT(compute_interpolation_coefficients)
  QCL_ENTRYPOINT(evaluate_lens_equation)
  QCL_MAKE_SOURCE(
    QCL_IMPORT_CONSTANT(Max_particles_per_ray)
    QCL_IMPORT_CONSTANT(Max_nodes_per_ray)
    QCL_INCLUDE_MODULE(spatialcl::configuration<type_system>)
    QCL_INCLUDE_MODULE(interpolation)
    QCL_INCLUDE_MODULE(pixel_screen)
    QCL_INCLUDE_MODULE(lensing_moments)
    QCL_IMPORT_TYPE(vector2)
    QCL_IMPORT_TYPE(vector8)
    QCL_IMPORT_CONSTANT(num_interpolation_cells)
    QCL_IMPORT_CONSTANT(interpolation_ray_grid_size)
    QCL_IMPORT_CONSTANT(num_multipole_evaluations)
    QCL_IMPORT_CONSTANT(secondary_rays_per_cell)
    QCL_IMPORT_CONSTANT(cells_per_ray)
    R"(

      #define LOAD_INTERPOLATION_VALUES(output_x, output_y,                 \
                                        temp_deflection,                    \
                                        deflections_begin,                  \
                                        offset_indices,                     \
                                        vector_element)                     \
        temp_deflection = deflections_begin[offset_indices.vector_element]; \
        output_x.vector_element = temp_deflection.s0;                       \
        output_y.vector_element = temp_deflection.s1


      __constant uchar16 interpolation_indices [] =
      {
        // Lower left interpolation cell
        (uchar16)(0,1,2,3, 5,6,7,8,     10,11,12,13, 15,16,17,18),
        // Lower right interpolation cell
        (uchar16)(1,2,3,4, 6,7,8,9,     11,12,13,14, 16,17,18,19),
        // Upper left interpolation cell
        (uchar16)(5,6,7,8, 10,11,12,13, 15,16,17,18, 20,21,22,23),
        // Upper right interpolation cell
        (uchar16)(6,7,8,9, 11,12,13,14, 16,17,18,19, 21,22,23,24)
      };
    )"
    QCL_RAW(

      __kernel void compute_interpolation_deflections(__global vector8* selected_nodes0,
                                                      __global vector8* selected_nodes1,
                                                      __global uint* num_selected_nodes,
                                                      __global vector2* primary_ray_positions,
                                                      ulong num_primary_rays,
                                                      scalar primary_ray_separation,
                                                      __global vector2* interpolation_deflections)
      {
        __local vector16 multipole_cache [Max_nodes_per_ray];

        const size_t lid = get_local_id(0);
        const int2 lid2d = (int2)(lid % interpolation_ray_grid_size - interpolation_ray_grid_size/2,
                                  lid / interpolation_ray_grid_size - interpolation_ray_grid_size/2);
        const scalar secondary_ray_separation = 0.5f * primary_ray_separation;

        for(size_t primary_ray_id = get_group_id(0);
            primary_ray_id < num_primary_rays;
            primary_ray_id += get_num_groups(0))
        {
          const vector2 primary_ray_position = primary_ray_positions[primary_ray_id];
          const uint num_nodes = num_selected_nodes[primary_ray_id];

          for(int i = lid; i < num_nodes; i += get_local_size(0))
          {
            lensing_multipole_expansion expansion;
            EXPANSION_LO(expansion) = selected_nodes0[primary_ray_id * Max_nodes_per_ray + i];
            EXPANSION_HI(expansion) = selected_nodes1[primary_ray_id * Max_nodes_per_ray + i];

            multipole_cache[i] = expansion;
          }

          barrier(CLK_LOCAL_MEM_FENCE);

          if(lid < num_multipole_evaluations)
          {
            const vector2 evaluation_point = primary_ray_position
                                           + secondary_ray_separation * convert_float2(lid2d);

            vector2 deflection = (vector2)0.0f;

            for(int i = 0; i < num_nodes; ++i)
              deflection -= multipole_expansion_evaluate(multipole_cache[i], evaluation_point);

            interpolation_deflections[primary_ray_id * num_multipole_evaluations + lid] = deflection;
          }
        }
      }



      __kernel void compute_interpolation_coefficients(
                      __global vector2* interpolation_deflections,
                      ulong num_primary_rays,
                      __global bicubic_interpolation_coefficients* coefficients_x,
                      __global bicubic_interpolation_coefficients* coefficients_y)
      {
        size_t tid = get_global_id(0);

        if(tid < cells_per_ray * num_primary_rays)
        {
          const ulong primary_ray_id = tid / cells_per_ray;
          vector16 input_values_x;
          vector16 input_values_y;

          const uchar16 cell_interpolation_indices = interpolation_indices[tid - primary_ray_id];
          __global vector2* deflections_begin =
                      interpolation_deflections + primary_ray_id * num_multipole_evaluations;

          vector2 temp;
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s0);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s1);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s2);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s3);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s4);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s5);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s6);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s7);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s8);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, s9);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, sa);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, sb);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, sc);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, sd);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, se);
          LOAD_INTERPOLATION_VALUES(input_values_x, input_values_y,
                                    temp, deflections_begin, cell_interpolation_indices, sf);

          const bicubic_interpolation_coefficients coeffs_x = bicubic_interpolation_init(input_values_x);
          const bicubic_interpolation_coefficients coeffs_y = bicubic_interpolation_init(input_values_y);

          coefficients_x[tid] = coeffs_x;
          coefficients_y[tid] = coeffs_y;
        }
      }

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
                        ray_id / secondary_rays_per_cell);
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
        const size_t primary_ray_id       = get_group_id(0) / cells_per_ray;
        const uchar interpolation_cell_id = get_group_id(0) & 3;
        const scalar interpolation_cell_size = primary_ray_separation * 0.5f;
        const scalar secondary_ray_separation = interpolation_cell_size / secondary_rays_per_cell;

        __local particle_type particle_cache [Max_particles_per_ray];

        const vector2 interpolation_cell_min_corner = primary_ray_positions[primary_ray_id]
                   + interpolation_cell_size * convert_float2(interpolation_cell_id2d(interpolation_cell_id));
        const uint num_particles = num_exact_particles[primary_ray_id];

        vector2 deflection = (vector2)0.0f;

        const uchar2 rid = ray_id2d(lid);
        const vector2 evaluation_position = interpolation_cell_min_corner
            + secondary_ray_separation*(convert_float2(rid)+(vector2)0.5f);


        // Collectively load exact particles into local memory
        if(lid < num_particles)
          particle_cache[lid] = exact_particles[primary_ray_id * Max_particles_per_ray + lid];

        barrier(CLK_LOCAL_MEM_FENCE);

        // First, calculate close-range deflections
        for(int i = 0; i < num_particles; ++i)
        {
          const particle_type p = particle_cache[i];
          const vector2 R = evaluation_position - p.xy;
          deflection += p.z * R * native_recip(dot(R,R));
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
        const size_t tid = get_global_id(0);

        if(tid < num_rays)
        {
          const ulong ray_id = tid + start_ray;
          // ToDo: Test tiling optimizations
          const uint rid_x = ray_id % num_rays_x;
          const uint rid_y = ray_id / num_rays_x;

          out[tid] = screen_min_corner
                   + (vector2)(rid_x, rid_y) * (vector2)(ray_separation, ray_separation);

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

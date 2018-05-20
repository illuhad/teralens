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

#ifndef LENSING_TREE_HPP
#define LENSING_TREE_HPP

#include <cstdlib>
#include <cassert>

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>
#include <QCL/qcl_array.hpp>

#include <SpatialCL/tree.hpp>
#include <SpatialCL/types.hpp>
#include <SpatialCL/configuration.hpp>
#include <SpatialCL/tree/binary_tree.hpp>

#include "configuration.hpp"
#include "lensing_moments.hpp"

namespace teralens {

using hilbert_sorter =
  spatialcl::key_based_sorter<
    spatialcl::hilbert_sort_key_generator<
      type_system
    >
  >;


using basic_lensing_tree = spatialcl::particle_tree
<
  hilbert_sorter, // Sort stars along a Hilbert curve
  type_system,
  vector8, // Use two 8-component vectors for each
  vector8  // node for the multipole expansion
>;

class lensing_tree : public basic_lensing_tree
{
public:
  QCL_MAKE_MODULE(lensing_tree)

  using vector_type =
    spatialcl::configuration<type_system>::vector_type;


  lensing_tree(const qcl::device_context_ptr& ctx,
               const qcl::device_array<particle_type>& particles)
      : basic_lensing_tree{ctx, particles}, _ctx{ctx}
  {
    // Need at least two particles for the tree
    assert(particles.size() > 2);

    this->init_multipoles();
  }

private:

  // Must be a power of 2
  static constexpr std::size_t reduction_group_size = 256;

  /// Initializes the tree nodes
  void init_multipoles()
  {
    assert(this->get_num_node_levels() > 0);
    // First, calculate center of masses and node widths.
    // This is done in a hierarchical fashion, i.e.,
    // the monopoles and node shapes in one level are calculated
    // from the monopoles and node shapes of the children.
    this->init_nodes_and_monopoles();
    // Calculate higher multipole moments
    this->init_higher_multipoles();
    // Wait for the construction to complete
    cl_int err = _ctx->get_command_queue().finish();
    qcl::check_cl_error(err, "Error while waiting for the tree node "
                             "construction to finish.");
  }

  /// Calculates monopole moments, center of masses, node widths
  /// and node centers for all nodes
  void init_nodes_and_monopoles()
  { 
    // Temporary storage for the center coordinates of the nodes
    qcl::device_array<vector2> node_centers{_ctx, this->get_effective_num_particles()};

    // First, build lowest level
    std::size_t lowest_level_num_nodes = (this->get_num_particles() + 1) >> 1;
    cl_int err = this->build_ll_nodes(_ctx,
                                      cl::NDRange{lowest_level_num_nodes},
                                      cl::NDRange{256})
          (this->get_node_values0(),
           this->get_node_values1(),
           this->get_sorted_particles(),
           static_cast<cl_ulong>(this->get_num_particles()),
           node_centers);

    qcl::check_cl_error(err, "Could not enqueue build_ll_nodes kernel");

    for(int level = this->get_num_node_levels()-2; level >= 0; --level)
    {
      // Build higher levels
      err = this->build_nodes(_ctx,
                              cl::NDRange{1ul << level},
                              cl::NDRange{256})
          (this->get_node_values0(),
           static_cast<cl_uint>(level),
           static_cast<cl_ulong>(this->get_num_particles()),
           static_cast<cl_ulong>(this->get_effective_num_particles()),
           static_cast<cl_ulong>(this->get_effective_num_levels()),
           node_centers);
      qcl::check_cl_error(err, "Could not enqueue build_nodes kernel");
    }

    // Set the node_extent field to the node center, because the extents
    // are not required anymore after the tree construction
    err = this->assign_node_centers_to_node_extents(_ctx,
                                                    cl::NDRange{this->get_num_nodes()},
                                                    cl::NDRange{128})(
            node_centers,
            this->get_node_values0(),
            static_cast<cl_ulong>(this->get_num_nodes()));
    qcl::check_cl_error(err, "Could not enqueue assign_node_extents_to_node_centers kernel");

    err = _ctx->get_command_queue().finish();
    qcl::check_cl_error(err,"Error while waiting for build_monopoles kernel to finish");
  }

  /// Initializes the higher multipole moments (from quadrupole to 64-pole)
  void init_higher_multipoles()
  {
    const std::size_t num_summation_groups =
            this->get_effective_num_particles() / reduction_group_size;

    qcl::device_array<vector2> reduction_spill_buffer0{_ctx, num_summation_groups};
    qcl::device_array<vector8> reduction_spill_buffer1{_ctx, num_summation_groups};


    std::size_t particles_per_node = 4;
    for(int level = this->get_num_node_levels()-2; level >= 0; --level)
    {
      this->reset_reduction_spill_buffers(reduction_spill_buffer0,
                                          reduction_spill_buffer1,
                                          num_summation_groups);

      cl_int err = this->build_higher_multipole_moments(_ctx,
                                                        this->get_num_particles(),
                                                        cl::NDRange{reduction_group_size})(
               this->get_sorted_particles(),
               this->get_node_values0(),
               this->get_node_values1(),
               static_cast<cl_uint>(level),
               static_cast<cl_ulong>(this->get_num_particles()),
               static_cast<cl_ulong>(this->get_effective_num_particles()),
               static_cast<cl_ulong>(this->get_effective_num_levels()),
               reduction_spill_buffer0,
               reduction_spill_buffer1);

      qcl::check_cl_error(err, "Could not enqueue build_higher_multipole_moments kernel");

      const std::size_t target_num_results =
          this->get_effective_num_particles() / particles_per_node;

      std::size_t current_num_results =
          this->get_effective_num_particles() / reduction_group_size;

      // As long as more reductions need to be done
      while(current_num_results > target_num_results)
      {
        const std::size_t summation_group_size =
            current_num_results / target_num_results;

        assert(current_num_results % 2 == 0);
        assert(target_num_results % 2 == 0 || target_num_results == 1);
        assert(summation_group_size % 2 == 0);

        const std::size_t effective_summation_size =
            std::min(summation_group_size, static_cast<std::size_t>(reduction_group_size));

        bool is_final_summation = summation_group_size <= reduction_group_size;

        err = this->sum_reduction_spill_buffer(_ctx,
                                               current_num_results,
                                               reduction_group_size)(
                 reduction_spill_buffer0,
                 reduction_spill_buffer1,
                 static_cast<cl_uint>(effective_summation_size),
                 static_cast<cl_int> (is_final_summation),
                 static_cast<cl_uint>(level),
                 static_cast<cl_uint>(this->get_effective_num_levels()),
                 static_cast<cl_ulong>(this->get_effective_num_particles()),
                 static_cast<cl_ulong>(current_num_results),
                 this->get_node_values0(),
                 this->get_node_values1());

        qcl::check_cl_error(err, "Could not enqueue sum_reduction_spill_buffer kernel");

        current_num_results /= effective_summation_size;

        if(current_num_results == 0)
          current_num_results = 1;
      }

      particles_per_node *= 2;
    }
  }

  void reset_reduction_spill_buffers(const qcl::device_array<vector2>& spill_buffer0,
                                     const qcl::device_array<vector8>& spill_buffer1,
                                     std::size_t spill_buffer_size) const
  {
    cl_int err = this->init_reduction_spill_buffers(_ctx,
                                                    cl::NDRange{spill_buffer_size},
                                                    cl::NDRange{256})(
          spill_buffer0,
          spill_buffer1,
          spill_buffer_size);
    qcl::check_cl_error(err, "Could not enqueue init_reduction_spill_buffers kernel");
  }


  QCL_ENTRYPOINT(build_ll_nodes)
  QCL_ENTRYPOINT(build_nodes)
  QCL_ENTRYPOINT(assign_node_centers_to_node_extents)
  QCL_ENTRYPOINT(build_higher_multipole_moments)
  QCL_ENTRYPOINT(sum_reduction_spill_buffer)
  QCL_ENTRYPOINT(init_reduction_spill_buffers)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(spatialcl::configuration<type_system>)
    QCL_INCLUDE_MODULE(spatialcl::tree_configuration<basic_lensing_tree>)
    QCL_INCLUDE_MODULE(spatialcl::binary_tree)
    QCL_INCLUDE_MODULE(lensing_moments)
    QCL_IMPORT_CONSTANT(reduction_group_size)
    QCL_RAW(

        __kernel void init_reduction_spill_buffers(__global vector_type* spill_buffer0,
                                                   __global node_type1* spill_buffer1,
                                                   ulong spill_buffer_size)
        {
          size_t tid = get_global_id(0);
          if(tid < spill_buffer_size)
          {
            spill_buffer0[tid] = (vector_type)0.0f;
            spill_buffer1[tid] = (node_type1)0.0f;
          }
        }

        __kernel void build_ll_nodes(__global node_type0* nodes0,
                                     __global node_type1* nodes1,
                                     __global particle_type* particles,
                                     const ulong num_particles,
                                     __global vector_type* node_centers)
        {
          ulong num_nodes = num_particles >> 1;
          if(num_particles & 1)
            ++num_nodes;

          for(ulong tid = get_global_id(0);
              tid < num_nodes;
              tid += get_global_size(0))
          {
            ulong left_particle_idx  = tid << 1;
            ulong right_particle_idx = left_particle_idx + 1;

            particle_type left_particle  = particles[left_particle_idx];
            particle_type right_particle = left_particle;
            // Make sure that the mass of the right particle is set to 0
            // in case it does not exist!
            right_particle.z = 0.f;

            if(right_particle_idx < num_particles)
              right_particle = particles[right_particle_idx];

            vector_type center_of_mass;
            center_of_mass  = left_particle.z  * left_particle.xy;
            center_of_mass += right_particle.z * right_particle.xy;

            scalar total_mass = left_particle.z + right_particle.z;

            center_of_mass /= total_mass;

            vector_type node_extent = fabs(left_particle.xy - right_particle.xy);
            scalar      node_width  = fmax(node_extent.x, node_extent.y);

            lensing_multipole_expansion expansion = multipole_expansion_for_particle(center_of_mass,
                                                                                     left_particle)
                                                  + multipole_expansion_for_particle(center_of_mass,
                                                                                     right_particle);
            CENTER_OF_MASS(expansion) = center_of_mass;
            MASS          (expansion) = total_mass;
            NODE_EXTENT   (expansion) = node_extent;
            NODE_WIDTH    (expansion) = node_width;

            nodes0[tid] = EXPANSION_LO(expansion);
            nodes1[tid] = EXPANSION_HI(expansion);

            node_centers[tid] = 0.5f * (left_particle.xy + right_particle.xy);

          }
        }

        // Build nodes on higher levels
        __kernel void build_nodes(__global node_type0* nodes0,
                                  const uint current_level,
                                  const ulong num_particles,
                                  const ulong effective_num_particles,
                                  const ulong effective_num_levels,
                                  __global vector_type* node_centers)
        {
          ulong num_nodes = BT_NUM_NODES(current_level);

          for(ulong tid = get_global_id(0);
              tid < num_nodes;
              tid += get_global_size(0))
          {
            binary_tree_key_t node_key;
            binary_tree_key_init(&node_key, current_level, tid);

            if(binary_tree_is_node_used(&node_key,
                                        effective_num_levels,
                                        num_particles))
            {
              binary_tree_key_t children_begin = binary_tree_get_children_begin(&node_key);
              binary_tree_key_t children_last  = binary_tree_get_children_last (&node_key);

              int right_child_exists = binary_tree_is_node_used(&children_last,
                                                                effective_num_levels,
                                                                num_particles);

              ulong effective_child_idx = binary_tree_key_encode_global_id(&children_begin,
                                                                           effective_num_levels)
                                        - effective_num_particles;

              node_type0 left_child_node = nodes0[effective_child_idx];
              node_type0 right_child_node = (node_type0)0.0f;
              CENTER_OF_MASS(right_child_node) = CENTER_OF_MASS(left_child_node);

              vector_type left_child_node_extent = NODE_EXTENT(left_child_node);
              vector_type right_child_node_extent = (vector_type)0.0f;

              vector_type left_child_node_center = node_centers[effective_child_idx];
              vector_type right_child_node_center = left_child_node_center;

              if(right_child_exists)
              {
                right_child_node        = nodes0[effective_child_idx + 1];
                right_child_node_extent = NODE_EXTENT(right_child_node);
                right_child_node_center = node_centers[effective_child_idx + 1];
              }
              scalar left_mass  = MASS(left_child_node );
              scalar right_mass = MASS(right_child_node);
              // Calculate center of mass
              vector_type parent_com =
                left_mass  * CENTER_OF_MASS(left_child_node)
              + right_mass * CENTER_OF_MASS(right_child_node);

              scalar total_mass = left_mass + right_mass;
              parent_com /= total_mass;

              vector_type parent_min_corner =
                  fmin(left_child_node_center  - 0.5f * NODE_EXTENT(left_child_node),
                       right_child_node_center - 0.5f * NODE_EXTENT(right_child_node));

              vector_type parent_max_corner =
                  fmax(left_child_node_center  + 0.5f * NODE_EXTENT(left_child_node),
                       right_child_node_center + 0.5f * NODE_EXTENT(right_child_node));

              node_type0 parent_node = (node_type0)0.0f;
              CENTER_OF_MASS(parent_node) = parent_com;
              MASS          (parent_node) = total_mass;
              NODE_EXTENT   (parent_node) = parent_max_corner - parent_min_corner;
              NODE_WIDTH    (parent_node) = fmax(NODE_EXTENT_X(parent_node),
                                                 NODE_EXTENT_Y(parent_node));
              /*NODE_WIDTH    (parent_node) = sqrt(NODE_EXTENT_X(parent_node)*NODE_EXTENT_X(parent_node) +
                                                 NODE_EXTENT_Y(parent_node)*NODE_EXTENT_Y(parent_node));*/

              ulong effective_node_idx = binary_tree_key_encode_global_id(&node_key,
                                                                          effective_num_levels)
                                       - effective_num_particles;
              // Set result
              nodes0      [effective_node_idx] = parent_node;
              node_centers[effective_node_idx] = 0.5f * (parent_min_corner + parent_max_corner);
            }
          }
        }

        __kernel void assign_node_centers_to_node_extents(__global vector_type* node_centers,
                                                          __global node_type0* nodes0,
                                                          ulong num_nodes)
        {
          size_t tid = get_global_id(0);

          if(tid < num_nodes)
            NODE_EXTENT(nodes0[tid]) = node_centers[tid];
        }

        void sum_multipoles(volatile __local vector_type* subgroup_mem_nodes0,
                            volatile __local node_type1*  subgroup_mem_nodes1,
                            const size_t subgroup_lid,
                            const size_t summation_size)
        {
          for(int i = summation_size/2; i > 0; i >>= 1)
          {
            if(subgroup_lid < i)
            {
              subgroup_mem_nodes0[subgroup_lid] = subgroup_mem_nodes0[subgroup_lid  ] +
                                                  subgroup_mem_nodes0[subgroup_lid+i];

              subgroup_mem_nodes1[subgroup_lid] = subgroup_mem_nodes1[subgroup_lid  ] +
                                                  subgroup_mem_nodes1[subgroup_lid+i];
            }
            // This can be optimized for small summations
            barrier(CLK_LOCAL_MEM_FENCE);
          }
        }

        __kernel void build_higher_multipole_moments(__global particle_type* particles,
                                                     __global node_type0* nodes0,
                                                     __global node_type1* nodes1,
                                                     const uint  current_level,
                                                     const ulong num_particles,
                                                     const ulong effective_num_particles,
                                                     const ulong effective_num_levels,
                                                     __global vector_type* reduction_spill_buffer0,
                                                     __global  node_type1* reduction_spill_buffer1)
        {
/*
          const ulong num_particles_per_node = BT_LEAVES_PER_NODE(current_level, effective_num_levels);
          size_t tid = get_global_id(0);
          const ulong effective_node_idx =
                  BT_LEVEL_OFFSET(current_level, effective_num_levels) - effective_num_particles
                  + tid;

          if(tid < BT_NUM_NODES(current_level))
          {
            node_type0 node0 = (node_type0)0.0f;
            node_type1 node1 = (node_type1)0.0f;

            vector_type center_of_mass = CENTER_OF_MASS(nodes0[effective_node_idx]);

            const ulong particles_begin = tid * num_particles_per_node;
            for(int i = 0; i < num_particles_per_node; ++i)
            {
              particle_type p = (particle_type)0.0f;
              if(particles_begin + i < num_particles)
                p = particles[particles_begin + i];


              lensing_multipole_expansion e =
                   multipole_expansion_for_particle(center_of_mass, p);
              QUADRUPOLE_MOMENT(node0) += QUADRUPOLE_MOMENT(e);
              node1 += EXPANSION_HI(e);
            }

            QUADRUPOLE_MOMENT(nodes0[effective_node_idx]) = QUADRUPOLE_MOMENT(node0);
            nodes1[effective_node_idx] = node1;
            //QUADRUPOLE_MOMENT(nodes0[effective_node_idx]) = (vector_type)0.0f;
            //nodes1[effective_node_idx] = (node_type1)0.0f;
          }*/

          // We only need to store the quadrupole moments from nodes0, this
          // fits into a 2d vector instead of a full-blown node_type0 vector.
          __local vector_type nodes0_local_mem [reduction_group_size];
          __local node_type1  nodes1_local_mem [reduction_group_size];

          particle_type current_particle = (particle_type)0.0f;

          const size_t tid = get_global_id(0);
          const size_t lid = get_local_id(0);
          lensing_multipole_expansion expansion = (lensing_multipole_expansion)0.0f;


          ulong num_particles_per_node = BT_LEAVES_PER_NODE(current_level,
                                                            effective_num_levels);
          binary_tree_key_t node_key;
          binary_tree_key_init(&node_key, current_level, tid / num_particles_per_node);
          ulong effective_node_idx = binary_tree_key_encode_global_id(&node_key,
                                                                      effective_num_levels)
                                   - effective_num_particles;

          if(tid < num_particles)
          {
            current_particle = particles[tid];

            vector_type center_of_mass = CENTER_OF_MASS(nodes0[effective_node_idx]);
            expansion = multipole_expansion_for_particle(center_of_mass, current_particle);
          }

          nodes0_local_mem [lid] = QUADRUPOLE_MOMENT(expansion);
          nodes1_local_mem [lid] = EXPANSION_HI     (expansion);

          barrier(CLK_LOCAL_MEM_FENCE);

          const size_t subgroup_size = min(num_particles_per_node,
                                           (ulong)reduction_group_size);
          const size_t subgroup_id  = lid / subgroup_size;
          const size_t subgroup_lid = lid - subgroup_id * subgroup_size;

          sum_multipoles(nodes0_local_mem + subgroup_id * subgroup_size,
                         nodes1_local_mem + subgroup_id * subgroup_size,
                         subgroup_lid,
                         subgroup_size);

          if(num_particles_per_node <= reduction_group_size)
          {
            // We can directly write the results to memory
            if(subgroup_lid == 0 && tid < num_particles)
            {
              QUADRUPOLE_MOMENT(nodes0[effective_node_idx]) =
                          nodes0_local_mem[subgroup_id * subgroup_size];
              nodes1[effective_node_idx] =
                          nodes1_local_mem[subgroup_id * subgroup_size];
            }
          }
          else if(lid == 0)
          {
            // The results must be reduced across multiple work groups
            reduction_spill_buffer0[get_group_id(0)] = nodes0_local_mem[0];
            reduction_spill_buffer1[get_group_id(0)] = nodes1_local_mem[0];
          }
        }

        __kernel void sum_reduction_spill_buffer(__global vector_type* reduction_spill_buffer0,
                                                 __global  node_type1* reduction_spill_buffer1,

                                                 // Should not be larger than the group size
                                                 const uint  summation_group_size,
                                                 const int   is_final_summation,
                                                 const uint  level,
                                                 const uint  effective_num_levels,
                                                 const ulong effective_num_particles,
                                                 const ulong num_summation_elements,
                                                 __global node_type0* nodes0,
                                                 __global node_type1* nodes1)
        {
          // Create local memory buffers for a fast on-chip summation
          __local vector_type nodes0_local_mem [reduction_group_size];
          __local node_type1  nodes1_local_mem [reduction_group_size];

          // Obtain thread ids
          const size_t tid = get_global_id(0);
          const size_t lid = get_local_id(0);

          // Load data collectively into local memory. If more threads
          // then needed are available, these set their entries to 0
          // (which will then have no effect on the summation)
          if(tid < num_summation_elements)
          {
            nodes0_local_mem[lid] = reduction_spill_buffer0[tid];
            nodes1_local_mem[lid] = reduction_spill_buffer1[tid];
          }
          else
          {
            nodes0_local_mem[lid] = (vector_type)0.0f;
            nodes1_local_mem[lid] = (node_type1)0.0f;
          }
          barrier (CLK_LOCAL_MEM_FENCE);

          // We further divide the work group into subgroups in order
          // to process as many summation groups as possible

          // subgroup_id is the id of the sumation group within the current
          // work group
          const uint subgroup_id = lid / summation_group_size;
          // subgroup_lid is the id of the thread within its subgroup/summation group
          const uint subgroup_lid = lid - subgroup_id * summation_group_size;
          // num_subgroups is the number of subgroups in the work group.
          const uint num_subgroups = get_local_size(0) / summation_group_size;

          // Perform parallel reduction for each subgroup in local memory
          sum_multipoles(nodes0_local_mem + subgroup_id * summation_group_size,
                         nodes1_local_mem + subgroup_id * summation_group_size,
                         subgroup_lid,
                         summation_group_size);

          if(is_final_summation)
          {
            // We are done with the summation, write the results to the nodes
            if (subgroup_lid == 0 && (tid < num_summation_elements))
            {
              const ulong level_offset = BT_LEVEL_OFFSET(level, effective_num_levels)
                                       - effective_num_particles;
              const ulong node_id = level_offset
                                  + get_group_id(0) * num_subgroups
                                  + subgroup_id;

              QUADRUPOLE_MOMENT(nodes0[node_id])
                              = nodes0_local_mem[lid];
              nodes1[node_id] = nodes1_local_mem[lid];
            }
          }
          else if(lid == 0)
          {
            // Prepare the reduction spill buffers for another run
            reduction_spill_buffer0[get_group_id(0)] = nodes0_local_mem[0];
            reduction_spill_buffer1[get_group_id(0)] = nodes1_local_mem[0];
          }
        }
    )
  )

  qcl::device_context_ptr _ctx;
};

}

#endif

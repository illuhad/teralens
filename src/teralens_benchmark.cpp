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

#include <iostream>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>
#include <random>

#include <boost/program_options.hpp>
#include <sstream>

#include "version.hpp"
#include "configuration.hpp"
#include "magnification_pattern.hpp"

constexpr std::size_t num_samples = 100;
constexpr std::size_t seed = 140;
constexpr std::size_t resolution = 1024;
constexpr teralens::scalar opening_angle = 0.4f;
constexpr teralens::scalar primary_rays_ppx = 0.1f;

int main()
{
  try
  {
    std::cout << "Teralens "
              << teralens::annotated_version_string()
              << ", Copyright (c) 2018 Aksel Alpay"
              << std::endl;


    qcl::environment env;

    const cl::Platform& platform =
        env.get_platform_by_preference({"NVIDIA", "AMD", "Intel"});
    qcl::global_context_ptr global_ctx =
        env.create_global_context(platform, CL_DEVICE_TYPE_GPU);

    if(global_ctx->get_num_devices() == 0)
    {
      std::cout << "No suitable OpenCL devices available." << std::endl;
      return -1;
    }

    qcl::device_context_ptr ctx = global_ctx->device(0);
    std::cout << "Using device: "     << ctx->get_device_name()
              << ", vendor: "         << ctx->get_device_vendor()
              << ", OpenCL version: " << ctx->get_device_cl_version()
              << std::endl;

    double total_runtime = 0.0f;
    std::size_t total_num_rays = 0;

    std::mt19937 random_engine(seed);
    std::uniform_real_distribution<teralens::scalar> random_distribution{0.0, 1.0};

    for(std::size_t i = 0; i < num_samples; ++i)
    {
      teralens::scalar total_convergence = random_distribution(random_engine);
      teralens::scalar smooth_fraction = random_distribution(random_engine);
      teralens::scalar shear = random_distribution(random_engine);

      teralens::scalar stellar_convergence = total_convergence * (1.f - smooth_fraction);
      teralens::scalar smooth_convergence = total_convergence * smooth_fraction;

      teralens::lensing_system system{
        ctx,
        1.0f, // mean particle mass
        stellar_convergence,
        smooth_convergence,
        shear,
        20.f, // source plane/magnification pattern size
        600001
      };

      std::cout << "N_* = " << system.get_particles().size()
                << " kappa_* = " << stellar_convergence
                << " kappa_smooth = " << smooth_convergence
                << " gamma = " << shear << ": ";

      std::stringstream dummy_output;
      teralens::magnification_pattern_generator generator{ctx, system, dummy_output};
      qcl::device_array<int> pixel_screen =
          generator.run(resolution, primary_rays_ppx, opening_angle);

      std::cout << 1.0/generator.get_last_runtime() << " patterns/s, "
                << generator.get_last_num_traced_rays()/generator.get_last_runtime() << " rays/s"
                << std::endl;

      total_runtime += generator.get_last_runtime();
      total_num_rays += generator.get_last_num_traced_rays();
    }
    std::cout << "====================================================" << std::endl;
    std::cout << "Average performance: "
              << num_samples/total_runtime << " patterns/s, "
              << total_num_rays/total_runtime << " rays/s"
              << std::endl;

  }
  catch(std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
    return -1;
  }
}

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

#include <boost/program_options.hpp>

#include "configuration.hpp"
#include "magnification_pattern.hpp"
#include "fits.hpp"
#include "multi_array.hpp"
#include "version.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv)
{
  try
  {
    std::cout << "Teralens "
              << teralens::annotated_version_string()
              << ", Copyright (c) 2018 Aksel Alpay"
              << std::endl;

    teralens::scalar stellar_convergence, smooth_convergence;
    teralens::scalar shear;
    teralens::scalar physical_source_plane_size;
    std::size_t random_seed;
    teralens::scalar tree_opening_angle;
    teralens::scalar primary_rays_ppx;

    std::size_t resolution;

    std::string output_filename;

    std::string rays_ppx_desc =
        "Number of primary rays traced for each pixel. "
        "The total number of rays is given by the total "
        "number of primary rays times "
        +std::to_string(teralens::num_interpolation_cells
                       *teralens::secondary_rays_per_cell
                       *teralens::secondary_rays_per_cell);

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "print help message")
        ("kappa_star",
         po::value<teralens::scalar>(&stellar_convergence)->default_value(0.2f),
         "convergence due to stars")
        ("kappa_smooth",
         po::value<teralens::scalar>(&smooth_convergence)->default_value(0.0f),
         "convergence due to smooth matter")
        ("gamma,g", po::value<teralens::scalar>(&shear)->default_value(0.1f), "shear")
        ("physical_size,s",
         po::value<teralens::scalar>(&physical_source_plane_size)->default_value(20.f),
         "The physical size of the magnification pattern in Einstein radii")
        ("seed", po::value<std::size_t>(&random_seed)->default_value(600001),
         "Random seed")
        ("opening_angle,a", po::value<teralens::scalar>(&tree_opening_angle)->default_value(0.4f),
         "Opening angle of Barnes-Hut tree")
        ("primary_rays_ppx,p", po::value<teralens::scalar>(&primary_rays_ppx)->default_value(0.1f),
         rays_ppx_desc.c_str())
        ("resolution,r", po::value<std::size_t>(&resolution)->default_value(1024),
         "Number of pixels of the magnification pattern in x and y direction")
        ("output,o", po::value<std::string>(&output_filename)->default_value("teralens.fits"),
         "Filename of output fits file")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 1;
    }

    if(primary_rays_ppx <= 0.0f)
      throw std::invalid_argument{"Number of primary rays per pixel must be greater than 0"};
    if(physical_source_plane_size <= 0.0)
      throw std::invalid_argument{"Size of magnification pattern must be greater than 0"};
    if(tree_opening_angle < 0.0f)
      throw std::invalid_argument{"Tree opening angle must be positive"};
    if(resolution == 0)
      throw std::invalid_argument{"Resolution must be greater than 0"};


    qcl::environment env;
#ifdef TERALENS_CPU_FALLBACK
    qcl::global_context_ptr global_ctx = env.create_global_cpu_context();
#else
    const cl::Platform& platform =
        env.get_platform_by_preference({"NVIDIA", "AMD", "Intel"});
    qcl::global_context_ptr global_ctx =
        env.create_global_context(platform, CL_DEVICE_TYPE_GPU);
#endif

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

    teralens::lensing_system system{
      ctx,
      1.0f, // mean particle mass
      stellar_convergence,
      smooth_convergence,
      shear,
      physical_source_plane_size, // source plane/magnification pattern size
      random_seed
    };

    std::cout << "Number of lenses: " << system.get_particles().size() << std::endl;


    teralens::magnification_pattern_generator generator{ctx, system, std::cout};
    qcl::device_array<int> pixel_screen =
        generator.run(resolution, primary_rays_ppx, tree_opening_angle);

    // Copy results back to the CPU
    teralens::util::multi_array<int> image{resolution, resolution};
    pixel_screen.read(image.data(), pixel_screen.begin(), pixel_screen.end());

    // Save results
    teralens::util::fits<int> output{output_filename};
    output.save(image);
  }
  catch(std::exception& e)
  {
    std::cout << "Error: " << e.what() << std::endl;
    return -1;
  }
}

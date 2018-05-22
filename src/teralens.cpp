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
#include <limits>
#include <memory>

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
    std::cout << "************************************************************\n";
    std::cout << "  Teralens " << teralens::annotated_version_string() << ",\n"
              << "   Copyright (c) 2018 Aksel Alpay\n"
              << "\n"
              << "  This program comes with ABSOLUTELY NO WARRANTY; It is\n"
              << "  free software, and you are welcome to redistribute it\n"
              << "  under the conditions of the GNU General Public License v3.\n";
    std::cout << "************************************************************\n";

    teralens::scalar stellar_convergence, smooth_convergence;
    teralens::scalar shear;
    teralens::scalar physical_source_plane_size;
    std::size_t random_seed;
    teralens::scalar tree_opening_angle;
    teralens::scalar primary_rays_ppx;
    teralens::scalar ray_sampling_region_scale;

    std::size_t resolution;

    std::string output_filename;
    std::string mode;
    std::string star_dump_output_filename;
    std::string star_dump_input_filename;

    std::string rays_ppx_desc =
        "Number of primary rays traced for each pixel. "
        "The total number of rays is given by the total "
        "number of primary rays times "
        +std::to_string(teralens::num_interpolation_cells
                       *teralens::secondary_rays_per_cell
                       *teralens::secondary_rays_per_cell);

    std::size_t device_id;
    std::string platform_vendor;

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
        ("opening_angle,a", po::value<teralens::scalar>(&tree_opening_angle)->default_value(0.5f),
         "Opening angle of Barnes-Hut tree")
        ("primary_rays_ppx,p", po::value<teralens::scalar>(&primary_rays_ppx)->default_value(0.1f),
         rays_ppx_desc.c_str())
        ("resolution,r", po::value<std::size_t>(&resolution)->default_value(1024),
         "Number of pixels of the magnification pattern in x and y direction")
        ("output,o", po::value<std::string>(&output_filename)->default_value("teralens.fits"),
         "Filename of output fits file")
        ("mode,m", po::value<std::string>(&mode)->default_value("auto"),
         "backend selection mode. Available options: tree, brute_force, auto")
        ("write_star_dump", po::value<std::string>(&star_dump_output_filename),
         "If set, the list of sampled stars will be saved to this file, with each row containing x coordinate, "
         "y coordinate, and mass.")
        ("read_star_dump", po::value<std::string>(&star_dump_input_filename),
         "If set, loads the stars from the specified file. The kappa_star argument is ignored in this "
         "case. Centers the magnification pattern on the center of mass of the stars.")
        ("ray_sampling_region_scale", po::value<teralens::scalar>(&ray_sampling_region_scale)->default_value(1.f),
         "Scaling factor for the ray sampling region")
        ("platform_vendor", po::value<std::string>(&platform_vendor),
         "The first OpenCL platform containing this string in its vendor identification will "
         "be selected for computation")
        ("device_id", po::value<std::size_t>(&device_id)->default_value(0),
         "The index of the used device within the eligible devices. If --platform_vendor is not set, "
         "this described the overall index of the GPUs (or CPUs) attached to the system. Otherwise,"
         " corresponds to the device index within the platform specified by the --platform_vendor flag.")
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

    std::size_t brute_force_threshold = 0;
    if(mode == "auto")
      brute_force_threshold = teralens::max_brute_force_lenses;
    else if(mode == "tree")
      brute_force_threshold = 0;
    else if(mode == "brute_force")
      brute_force_threshold = std::numeric_limits<std::size_t>::max();
    else
      throw std::invalid_argument{"Unkown backend mode: "+mode};



    qcl::environment env;
    qcl::global_context_ptr global_ctx;
#ifdef TERALENS_CPU_FALLBACK
    if(platform_vendor.empty())
      global_ctx = env.create_global_cpu_context();
    else
    {
      const cl::Platform& platform = env.get_platform_by_preference({platform_vendor});
      global_ctx = env.create_global_context(platform, CL_DEVICE_TYPE_CPU);
    }
#else
    if(platform_vendor.empty())
      global_ctx = env.create_global_gpu_context();
    else
    {
      const cl::Platform& platform =
          env.get_platform_by_preference({platform_vendor});
      global_ctx = env.create_global_context(platform, CL_DEVICE_TYPE_GPU);
    }
#endif

    if(global_ctx->get_num_devices() == 0)
    {
      std::cout << "No suitable OpenCL devices available." << std::endl;
      return -1;
    }

    std::cout << "Suitable devices:" << std::endl;
    for(std::size_t dev = 0; dev < global_ctx->get_num_devices(); ++dev)
    {
      std::cout << " " << dev << ": " << global_ctx->device(dev)->get_device_name()
                << " (" << global_ctx->device(dev)->get_device_vendor()
                << ", " << global_ctx->device(dev)->get_device_cl_version()<< ")";
      if(dev == device_id)
        std::cout << " [selected]";
      std::cout << std::endl;
    }

    if(device_id >= global_ctx->get_num_devices())
    {
      std::cout << "No device with requested index " << device_id
                << " exists within the suitable set of devices."
                << std::endl;
      return -1;
    }

    qcl::device_context_ptr ctx = global_ctx->device(device_id);
    std::cout << "Using device: "     << ctx->get_device_name()
              << ", vendor: "         << ctx->get_device_vendor()
              << ", OpenCL version: " << ctx->get_device_cl_version()
              << std::endl;

    ctx->enable_fast_relaxed_math();

    using system_ptr = std::unique_ptr<teralens::lensing_system>;
    system_ptr system;

    if(!star_dump_input_filename.empty())
    {
      system = system_ptr{
        new teralens::lensing_system{
          ctx,
          star_dump_input_filename,
          smooth_convergence,
          shear,
          physical_source_plane_size,
          ray_sampling_region_scale
        }
      };
    }
    else
    {
      system = system_ptr{
        new teralens::lensing_system{
          ctx,
          1.0f, // mean particle mass
          stellar_convergence,
          smooth_convergence,
          shear,
          physical_source_plane_size, // source plane/magnification pattern size
          random_seed,
          ray_sampling_region_scale
        }
      };
    }

    std::cout << "Number of lenses: "
              << system->get_particles().size()
              << " (kappa_star = " << system->get_compact_convergence()
              << ")" << std::endl;

    if(!star_dump_output_filename.empty())
      system->write_star_dump(star_dump_output_filename);


    teralens::magnification_pattern_generator generator{ctx, *system, std::cout};
    qcl::device_array<int> pixel_screen =
        generator.run(resolution, primary_rays_ppx, tree_opening_angle, brute_force_threshold);

    std::cout << "Performance: "
              << 1.0/generator.get_last_runtime() << " patterns/s, "
              << generator.get_last_num_traced_rays()/generator.get_last_runtime() << " rays/s"
              << std::endl;

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

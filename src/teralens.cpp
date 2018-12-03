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
#include <sstream>
#include <fstream>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>

#include <boost/program_options.hpp>

#include "configuration.hpp"
#include "magnification_pattern.hpp"
#include "fits.hpp"
#include "multi_array.hpp"
#include "version.hpp"

namespace po = boost::program_options;
#ifdef TERALENS_CPU_FALLBACK
constexpr std::size_t num_performance_estimates = 1;
#else
constexpr std::size_t num_performance_estimates = 10;
#endif

using system_ptr = std::unique_ptr<teralens::lensing_system>;

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

    // Define input parameters for the command line interface
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
    std::string cl_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "print help message")
        ("kappa-star",
         po::value<teralens::scalar>(&stellar_convergence)->default_value(0.2f),
         "convergence due to stars")
        ("kappa-smooth",
         po::value<teralens::scalar>(&smooth_convergence)->default_value(0.0f),
         "convergence due to smooth matter")
        ("gamma,g", po::value<teralens::scalar>(&shear)->default_value(0.1f), "shear")
        ("physical-size,s",
         po::value<teralens::scalar>(&physical_source_plane_size)->default_value(20.f),
         "The physical size of the magnification pattern in Einstein radii")
        ("seed", po::value<std::size_t>(&random_seed)->default_value(600001),
         "Random seed")
        ("opening-angle,a", po::value<teralens::scalar>(&tree_opening_angle)->default_value(0.5f),
         "Opening angle of Barnes-Hut tree")
        ("primary-rays-ppx,p", po::value<teralens::scalar>(&primary_rays_ppx)->default_value(0.1f),
         rays_ppx_desc.c_str())
        ("resolution,r", po::value<std::size_t>(&resolution)->default_value(1024),
         "Number of pixels of the magnification pattern in x and y direction")
        ("output,o", po::value<std::string>(&output_filename)->default_value("teralens.fits"),
         "Filename of output fits file")
        ("mode,m", po::value<std::string>(&mode)->default_value("auto"),
         "backend selection mode. Available options: tree, brute_force, auto")
        ("write-star-dump", po::value<std::string>(&star_dump_output_filename),
         "If set, the list of sampled stars will be saved to this file, with each row containing x coordinate, "
         "y coordinate, and mass.")
        ("read-star-dump", po::value<std::string>(&star_dump_input_filename),
         "If set, loads the stars from the specified file. The kappa_star argument is ignored in this "
         "case. Centers the magnification pattern on the center of mass of the stars.")
        ("ray-sampling-region-scale", po::value<teralens::scalar>(&ray_sampling_region_scale)->default_value(1.f),
         "Scaling factor for the ray sampling region")
        ("platform-vendor", po::value<std::string>(&platform_vendor),
         "The first OpenCL platform containing this string in its vendor identification will "
         "be selected for computation")
        ("device-id", po::value<std::size_t>(&device_id)->default_value(0),
         "The index of the used device within the eligible devices. If --platform-vendor is not set, "
         "this described the overall index of the GPUs (or CPUs) attached to the system. Otherwise,"
         " corresponds to the device index within the platform specified by the --platform-vendor flag.")
        ("disable-relaxed-math","If set, do not pass the -cl-fast-relaxed-math argument to the OpenCL compiler. "
                                "Usually, this is not required, as the -cl-fast-relaxed-math flag is typically "
                                "safe to use.")
        ("cl-options", po::value<std::string>(&cl_options), "Additional options for the OpenCL compiler")
        ("performance-measurement-mode", "If set, calculations will be carried out several times and the "
                                         "reported performance metrics will be the average of the different "
                                         "runs.")
        ;

    // Evaluate command line
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 1;
    }

    // Check sanity of arguments
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


    // Initialize OpenCL backend
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
    std::cout << std::endl;

    if(device_id >= global_ctx->get_num_devices())
    {
      std::cout << "No device with requested index " << device_id
                << " exists within the suitable set of devices."
                << std::endl;
      return -1;
    }

    qcl::device_context_ptr ctx = global_ctx->device(device_id);

    // We enable relaxed math unless it was explicitly disabled via
    // the --disable-relaxed-math command line argument
    if(!vm.count("disable-relaxed-math"))
    {
      std::cout << "Fast, relaxed math is enabled." << std::endl;
      ctx->enable_fast_relaxed_math();
    }
    if(!cl_options.empty())
    {
      ctx->append_build_option(cl_options);
      std::cout << "Using OpenCL compiler options: "
                << ctx->get_build_options()
                << std::endl;
    }
    bool benchmark_mode = false;
    if(vm.count("performance-measurement-mode"))
      benchmark_mode = true;


    system_ptr system;

    // If a star dump input filename was given, load stars from this file.
    // Otherwise, we generate a new sample of stars from the given
    // kappa_star value.
    // The constructor of teralens::lensing_system will either generate a star sample
    // or read from the star dump file
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

    // If an output filename for a star dump was given,
    // save the star list used in the subsequent calculation
    if(!star_dump_output_filename.empty())
      system->write_star_dump(star_dump_output_filename);

    qcl::device_array<int> pixel_screen;
    if(!benchmark_mode)
    {
      // Now the action begins!
      // ...first construct magnification_pattern_generator object
      teralens::magnification_pattern_generator generator{ctx, *system, std::cout};
      // ...and execute the calculation. We end up with a pixel_screen object
      // containing the ray counts for each pixel
      pixel_screen =
        generator.run(resolution, primary_rays_ppx, tree_opening_angle, brute_force_threshold);

      std::cout << "Performance: "
                << 1.0/generator.get_last_runtime() << " patterns/s, "
                << generator.get_last_num_traced_rays()/generator.get_last_runtime() << " rays/s"
                << std::endl;
    }
    else
    {
      // In benchmark mode, average performance over several runs and print detailed
      // performance metrics
      double total_time = 0.0;
      std::size_t total_num_rays = 0;
      std::string benchmark_log_file = output_filename + ".perf.dat";
      std::ofstream performance_log{benchmark_log_file.c_str(), std::ios::trunc};

      for(std::size_t i = 0; i < num_performance_estimates; ++i)
      {
        std::cout << "Processing performance measurement " << i << "..." << std::endl;
        teralens::magnification_pattern_generator generator{ctx, *system, std::cout};
        pixel_screen =
            generator.run(resolution, primary_rays_ppx, tree_opening_angle, brute_force_threshold);
        total_time += generator.get_last_runtime();
        total_num_rays += generator.get_last_num_traced_rays();
      }
      double mean_time = total_time / num_performance_estimates;
      std::stringstream output;
      output << "#N_*\t t_tot [s]\t rays_tot\t t_pattern [s]\t patterns/s\t rays/s\t" << std::endl;
      output << system->get_particles().size() << "\t "
             << total_time << "\t " << total_num_rays
             << "\t " << mean_time << "\t " << 1.0/mean_time << "\t "
             << total_num_rays / total_time
             << std::endl;
      std::cout << output.str();
      performance_log << output.str();

    }
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

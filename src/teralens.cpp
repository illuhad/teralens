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

#include "magnification_pattern.hpp"
#include "fits.hpp"
#include "multi_array.hpp"

int main(int argc, char** argv)
{
  std::cout << "Teralens version 0.1, Copyright (c) 2018 Aksel Alpay"
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

  teralens::lensing_system system{
    ctx,
    1.0f, // mean particle mass
    0.3f, // convergence_stars
    0.1f, // convergence_smooth
    0.1f, // shear
    20.0f, // source plane/magnification pattern size
    600001 // seed
  };

  std::size_t resolution = 1024;

  teralens::magnification_pattern_generator generator{ctx, system};
  qcl::device_array<int> pixel_screen = generator.run(resolution, 10);

  // Copy results back to the CPU
  teralens::util::multi_array<int> image{resolution, resolution};
  pixel_screen.read(image.data(), pixel_screen.begin(), pixel_screen.end());

  // Save results
  teralens::util::fits<int> output{"teralens.fits"};
  output.save(image);

}

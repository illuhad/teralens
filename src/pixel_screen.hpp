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

#ifndef PIXEL_SCREEN_HPP
#define PIXEL_SCREEN_HPP

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>

#include "configuration.hpp"

namespace teralens {

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

      _center{pixel_screen_center},
      _extent{pixel_screen_extent},

      _pixel_screen{ctx, num_pix_x * num_pix_y}
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

        int2 pixel = convert_int2_rtn((pos - screen_min_corner)/pixel_size);

        if(pixel.x >= 0 &&
           pixel.y >= 0 &&
           pixel.x < num_px_x &&
           pixel.y < num_px_y)
          atomic_inc(pixels + pixel.y * num_px_x + pixel.x);
      }

    )
  )
};

}

#endif

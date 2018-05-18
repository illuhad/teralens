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

#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <ctime>

namespace teralens {
  
class timer
{
public:
  timer()
  : _is_running{false}
  {}

  inline
  bool is_running() const 
  {return _is_running;}

  void start()
  {
    _is_running = true;
    _start = std::chrono::high_resolution_clock::now();
  }

  double stop()
  {
    if(!_is_running)
      return 0.0;

    _stop = std::chrono::high_resolution_clock::now();
    _is_running = false;

    auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(_stop - _start).count();
    return static_cast<double>(ticks) * 1.e-9;
  }

private:
  using time_point_type = 
    std::chrono::time_point<std::chrono::high_resolution_clock>;
  time_point_type _start;
  time_point_type _stop;

  bool _is_running;
};

}

#endif

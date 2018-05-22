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

#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>

#include "configuration.hpp"
#include "instructions.hpp"

namespace teralens {

using bicubic_interpolation_coefficients = vector16;
QCL_STANDALONE_MODULE(interpolation)
QCL_STANDALONE_SOURCE(
  QCL_INCLUDE_MODULE(instructions)
  QCL_IMPORT_TYPE(bicubic_interpolation_coefficients)
  QCL_IMPORT_TYPE(vector16)
  QCL_IMPORT_TYPE(vector2)
  R"(
    // Define access macros for the entries
    // of the interpolation matrix
    #define IP_M00(m) m.s0
    #define IP_M01(m) m.s1
    #define IP_M02(m) m.s2
    #define IP_M03(m) m.s3

    #define IP_M10(m) m.s4
    #define IP_M11(m) m.s5
    #define IP_M12(m) m.s6
    #define IP_M13(m) m.s7

    #define IP_M20(m) m.s8
    #define IP_M21(m) m.s9
    #define IP_M22(m) m.sa
    #define IP_M23(m) m.sb

    #define IP_M30(m) m.sc
    #define IP_M31(m) m.sd
    #define IP_M32(m) m.se
    #define IP_M33(m) m.sf

  )"

  QCL_RAW(
    bicubic_interpolation_coefficients bicubic_interpolation_init(const vector16 p)
    {
      bicubic_interpolation_coefficients a;
      IP_M00(a) =         IP_M11(p);
      IP_M01(a) = -0.5f * IP_M10(p) + 0.5f * IP_M12(p);
      IP_M02(a) =         IP_M10(p) - 2.5f * IP_M11(p) + 2.f  * IP_M12(p) - 0.5f * IP_M13(p);
      IP_M03(a) = -0.5f * IP_M10(p) + 1.5f * IP_M11(p) - 1.5f * IP_M12(p) + 0.5f * IP_M13(p);
      IP_M10(a) = -0.5f * IP_M01(p) + 0.5f * IP_M21(p);
      IP_M11(a) = 0.25f * IP_M00(p) - 0.25f* IP_M02(p) - 0.25f* IP_M20(p) + 0.25f* IP_M22(p);
      IP_M12(a) = -0.5f * IP_M00(p) + 1.25f* IP_M01(p) -        IP_M02(p) + 0.25f* IP_M03(p)
                  +0.5f * IP_M20(p) - 1.25f* IP_M21(p) +        IP_M22(p) - 0.25f* IP_M23(p);
      IP_M13(a) = 0.25f * IP_M00(p) - 0.75f* IP_M01(p) + 0.75f* IP_M02(p) - 0.25f* IP_M03(p)
                 -0.25f * IP_M20(p) + 0.75f* IP_M21(p) - 0.75f* IP_M22(p) + 0.25f* IP_M23(p);
      IP_M20(a) =         IP_M01(p) - 2.5f * IP_M11(p) + 2.f  * IP_M21(p) - 0.5f * IP_M31(p);
      IP_M21(a) = -0.5f * IP_M00(p) + 0.5f * IP_M02(p) + 1.25f* IP_M10(p) - 1.25f* IP_M12(p)
                        - IP_M20(p) +        IP_M22(p) + 0.25f* IP_M30(p) - 0.25f* IP_M32(p);
      IP_M22(a) =         IP_M00(p) - 2.5f * IP_M01(p) + 2    * IP_M02(p) - 0.5f * IP_M03(p)
                  -2.5f * IP_M10(p) + 6.25f* IP_M11(p) - 5    * IP_M12(p) + 1.25f* IP_M13(p)
                  +2    * IP_M20(p) - 5    * IP_M21(p) + 4    * IP_M22(p) -        IP_M23(p)
                  -0.5f * IP_M30(p) + 1.25f* IP_M31(p) -        IP_M32(p) + 0.25f* IP_M33(p);
      IP_M23(a) = -0.5f * IP_M00(p) + 1.5f * IP_M01(p) - 1.5f * IP_M02(p) + 0.5f * IP_M03(p)
                  +1.25f* IP_M10(p) - 3.75f* IP_M11(p) + 3.75f* IP_M12(p) - 1.25f* IP_M13(p)
                  -       IP_M20(p) + 3    * IP_M21(p) - 3    * IP_M22(p) +        IP_M23(p)
                  +0.25f* IP_M30(p) - 0.75f* IP_M31(p) + 0.75f* IP_M32(p) - 0.25f* IP_M33(p);
      IP_M30(a) = -0.5f * IP_M01(p) + 1.5f * IP_M11(p) - 1.5f * IP_M21(p) + 0.5f * IP_M31(p);
      IP_M31(a) =  0.25f* IP_M00(p) - 0.25f* IP_M02(p) - 0.75f* IP_M10(p) + 0.75f* IP_M12(p)
                  +0.75f* IP_M20(p) - 0.75f* IP_M22(p) - 0.25f* IP_M30(p) + 0.25f* IP_M32(p);
      IP_M32(a) = -0.5f * IP_M00(p) + 1.25f* IP_M01(p) -        IP_M02(p) + 0.25f* IP_M03(p)
                  +1.5f * IP_M10(p) - 3.75f* IP_M11(p) + 3   *  IP_M12(p) - 0.75f* IP_M13(p)
                  -1.5f * IP_M20(p) + 3.75f* IP_M21(p) - 3   *  IP_M22(p) + 0.75f* IP_M23(p)
                  +0.5f * IP_M30(p) - 1.25f* IP_M31(p) +        IP_M32(p) - 0.25f* IP_M33(p);
      IP_M33(a) =  0.25f* IP_M00(p) - 0.75f* IP_M01(p) + 0.75f* IP_M02(p) - 0.25f* IP_M03(p)
                  -0.75f* IP_M10(p) + 2.25f* IP_M11(p) - 2.25f* IP_M12(p) + 0.75f* IP_M13(p)
                  +0.75f* IP_M20(p) - 2.25f* IP_M21(p) + 2.25f* IP_M22(p) - 0.75f* IP_M23(p)
                  -0.25f* IP_M30(p) + 0.75f* IP_M31(p) - 0.75f* IP_M32(p) + 0.25f* IP_M33(p);
      return a;
    }

    scalar bicubic_unit_square_interpolation(const bicubic_interpolation_coefficients a,
                                             const vector2 relative_position)
    {
      const vector2 pos1 = relative_position;
      const vector2 pos2 = pos1 * pos1;
      const vector2 pos3 = pos2 * pos1;

      // Terms for x^0
      scalar result = IP_M00(a);
      result = FMA(IP_M01(a), pos1.y, result);
      result = FMA(IP_M02(a), pos2.y, result);
      result = FMA(IP_M03(a), pos3.y, result);

      // Terms for x^1
      scalar temp_result = IP_M10(a);
      temp_result = FMA(IP_M11(a), pos1.y, temp_result);
      temp_result = FMA(IP_M12(a), pos2.y, temp_result);
      temp_result = FMA(IP_M13(a), pos3.y, temp_result);
      result      = FMA(pos1.x, temp_result, result);

      // Terms for x^2
      temp_result  = IP_M20(a);
      temp_result = FMA(IP_M21(a), pos1.y, temp_result);
      temp_result = FMA(IP_M12(a), pos2.y, temp_result);
      temp_result = FMA(IP_M13(a), pos3.y, temp_result);
      result      = FMA(pos2.x, temp_result, result);

      // Terms for x^3
      temp_result  = IP_M30(a);
      temp_result = FMA(IP_M31(a), pos1.y, temp_result);
      temp_result = FMA(IP_M32(a), pos2.y, temp_result);
      temp_result = FMA(IP_M33(a), pos3.y, temp_result);
      result      = FMA(pos3.x, temp_result, result);

      return result;
    }
  )
)

}

#endif

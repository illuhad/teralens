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


/// \brief This file implements the multipole expansion of the deflection angle
/// due to gravitational lensing. For the derivation and formulae used, see
/// Wambsganß, J.: Gravitational microlensing.,
/// Max-Planck-Institut für Physik und Astrophysik, Garching (Germany). Inst. für Astrophysik,
/// Oct 1990


#ifndef LENSING_MOMENTS_HPP
#define LENSING_MOMENTS_HPP

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>
#include <SpatialCL/types.hpp>

#include "configuration.hpp"

namespace teralens {

using lensing_multipole_expansion = spatialcl::cl_vector_type<scalar, 16>::value;

class lensing_moments
{
public:
  QCL_MAKE_MODULE(lensing_moments)

private:
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(spatialcl::configuration<type_system>)
    QCL_IMPORT_TYPE(lensing_multipole_expansion)
    QCL_RAW(
      $pp define MULTIPOLE_ORDER 6 $

      // first half (lo) is stored in nodes0
      $pp define      CENTER_OF_MASS(expansion) expansion.s01 $
      $pp define    CENTER_OF_MASS_X(expansion) expansion.s0  $
      $pp define    CENTER_OF_MASS_Y(expansion) expansion.s1  $
      $pp define                MASS(expansion) expansion.s2  $
      $pp define     MONOPOLE_MOMENT(expansion) expansion.s2  $
      $pp define       NODE_EXTENT_X(expansion) expansion.s3  $
      $pp define       NODE_EXTENT_Y(expansion) expansion.s4  $
      $pp define         NODE_EXTENT(expansion) expansion.s34 $
      $pp define          NODE_WIDTH(expansion) expansion.s5  $
      $pp define QUADRUPOLE_MOMENT_X(expansion) expansion.s6  $
      $pp define QUADRUPOLE_MOMENT_Y(expansion) expansion.s7  $
      $pp define   QUADRUPOLE_MOMENT(expansion) expansion.s67 $

      // The second half of the expansion (hi) is stored in nodes1
      $pp define OCTOPOLE_MOMENT_X(expansion) expansion.s8 $
      $pp define OCTOPOLE_MOMENT_Y(expansion) expansion.s9 $
      $pp define   POLE16_MOMENT_X(expansion) expansion.sa $
      $pp define   POLE16_MOMENT_Y(expansion) expansion.sb $
      $pp define   POLE32_MOMENT_X(expansion) expansion.sc $
      $pp define   POLE32_MOMENT_Y(expansion) expansion.sd $
      $pp define   POLE64_MOMENT_X(expansion) expansion.se $
      $pp define   POLE64_MOMENT_Y(expansion) expansion.sf $


      $pp define EXPANSION_LO(expansion) expansion.s01234567 $
      $pp define EXPANSION_HI(expansion) expansion.s89abcdef $

      $pp  define PARTICLE_MASS(particle) particle.z $




      lensing_multipole_expansion multipole_expansion_for_particle(vector_type center_of_mass,
                                                                   particle_type p)
      {
        lensing_multipole_expansion result = (lensing_multipole_expansion)0.0f;

        // Quadrupole
$pp if MULTIPOLE_ORDER >= 2 $
        vector_type delta = PARTICLE_POSITION(p) - center_of_mass;

        vector_type delta_pow2 = delta      * delta;
        vector_type delta_pow3 = delta      * delta_pow2;
        vector_type delta_pow4 = delta_pow2 * delta_pow2;
        vector_type delta_pow5 = delta_pow2 * delta_pow3;
        vector_type delta_pow6 = delta_pow3 * delta_pow3;

        QUADRUPOLE_MOMENT_X(result) = delta_pow2.x - delta_pow2.y;
        QUADRUPOLE_MOMENT_Y(result) = 2 * delta.x * delta.y;
$pp endif $

        // OCTOPOLE
$pp if MULTIPOLE_ORDER >= 3 $
        OCTOPOLE_MOMENT_X(result) =  delta_pow3.x - 3 * delta.x * delta_pow2.y;
        OCTOPOLE_MOMENT_Y(result) = -delta_pow3.y + 3 * delta.y * delta_pow2.x;
$pp endif $

        // 16-pole
$pp if MULTIPOLE_ORDER >= 4 $
        POLE16_MOMENT_X(result) = delta_pow4.x - 6 * delta_pow2.x * delta_pow2.y + delta_pow4.y;
        POLE16_MOMENT_Y(result) = 4 * delta_pow3.x * delta.y - 4 * delta_pow3.y * delta.x;
$pp endif $

        // 32-pole
$pp if MULTIPOLE_ORDER >= 5 $
        POLE32_MOMENT_X(result) = delta_pow5.x - 10 * delta_pow3.x * delta_pow2.y + 5 * delta.x * delta_pow4.y;
        POLE32_MOMENT_Y(result) = delta_pow5.y - 10 * delta_pow3.y * delta_pow2.x + 5 * delta.y * delta_pow4.x;
$pp endif $

        // 64-pole
$pp if MULTIPOLE_ORDER >= 6 $
        POLE64_MOMENT_X(result) =
             delta_pow6.x - 15 * delta_pow4.x * delta_pow2.y + 15 * delta_pow2.x * delta_pow4.y - delta_pow6.y;
        POLE64_MOMENT_Y(result) =
             6 * delta_pow5.x * delta.y - 20 * delta_pow3.x * delta_pow3.y + 6 * delta_pow5.y * delta.x;
$pp endif $
        return PARTICLE_MASS(p) * result;
      }


      vector_type multipole_expansion_evaluate(const lensing_multipole_expansion m,
                                               const vector_type position)
      {
        vector_type result = (vector_type)(0,0);
        const vector_type R = position - CENTER_OF_MASS(m);

        const scalar r2 = dot(R, R);
        const scalar r2_inv = 1.f / r2;

        // monopole
        result = -MASS(m) * R * r2_inv;

        // Quadrupole (Dipole vanishes always)
$pp if MULTIPOLE_ORDER >= 2 $
        const vector_type R_pow2 = R      * R;
        const vector_type R_pow3 = R      * R_pow2;
        const vector_type R_pow4 = R_pow2 * R_pow2;
        const vector_type R_pow5 = R_pow2 * R_pow3;
        const vector_type R_pow6 = R_pow3 * R_pow3;
        const vector_type R_pow7 = R_pow3 * R_pow4;

        scalar r_pow2i_plus2_inv = r2_inv * r2_inv * r2_inv;
        vector_type evaluation_coefficients;
        scalar m0c0;
        scalar m1c1;
        scalar m1c0;
        scalar m0c1;

        evaluation_coefficients.x = r_pow2i_plus2_inv * (    R_pow3.x       - 3*R.x * R_pow2.y);
        evaluation_coefficients.y = r_pow2i_plus2_inv * (3 * R_pow2.x * R.y -         R_pow3.y);

        m0c0 = evaluation_coefficients.x * QUADRUPOLE_MOMENT_X(m);
        m1c1 = evaluation_coefficients.y * QUADRUPOLE_MOMENT_Y(m);
        m1c0 = evaluation_coefficients.x * QUADRUPOLE_MOMENT_Y(m);
        m0c1 = evaluation_coefficients.y * QUADRUPOLE_MOMENT_X(m);

        result.x -= (m0c0 + m1c1);
        result.y -= (m0c1 - m1c0);
$pp endif $

        // Octopole
$pp if MULTIPOLE_ORDER >= 3 $
        r_pow2i_plus2_inv *= r2_inv;

        evaluation_coefficients.x =
          r_pow2i_plus2_inv * (    R_pow4.x       - 6 * R_pow2.x * R_pow2.y + R_pow4.y);
        evaluation_coefficients.y =
          r_pow2i_plus2_inv * (4 * R_pow3.x * R.y - 4 * R.x      * R_pow3.y);

        m0c0 = evaluation_coefficients.x * OCTOPOLE_MOMENT_X(m);
        m1c1 = evaluation_coefficients.y * OCTOPOLE_MOMENT_Y(m);
        m1c0 = evaluation_coefficients.x * OCTOPOLE_MOMENT_Y(m);
        m0c1 = evaluation_coefficients.y * OCTOPOLE_MOMENT_X(m);

        result.x -= (m0c0 + m1c1);
        result.y -= (m0c1 - m1c0);
$pp endif $

        // 16-pole
$pp if MULTIPOLE_ORDER >= 4 $
        r_pow2i_plus2_inv *= r2_inv;

        evaluation_coefficients.x =
                r_pow2i_plus2_inv * (    R_pow5.x       - 10 * R_pow3.x * R_pow2.y + 5 * R.x * R_pow4.y);
        evaluation_coefficients.y =
                r_pow2i_plus2_inv * (5 * R_pow4.x * R.y - 10 * R_pow2.x * R_pow3.y +           R_pow5.y);

        m0c0 = evaluation_coefficients.x * POLE16_MOMENT_X(m);
        m1c1 = evaluation_coefficients.y * POLE16_MOMENT_Y(m);
        m1c0 = evaluation_coefficients.x * POLE16_MOMENT_Y(m);
        m0c1 = evaluation_coefficients.y * POLE16_MOMENT_X(m);

        result.x -= (m0c0 + m1c1);
        result.y -= (m0c1 - m1c0);
$pp endif $

        // 32-pole
$pp if MULTIPOLE_ORDER >= 5 $
        r_pow2i_plus2_inv *= r2_inv;

        evaluation_coefficients.x =
          r_pow2i_plus2_inv * (    R_pow6.x       - 15 * R_pow4.x * R_pow2.y + 15 * R_pow2.x * R_pow4.y - R_pow6.y);
        evaluation_coefficients.y =
          r_pow2i_plus2_inv * (6 * R_pow5.x * R.y - 20 * R_pow3.x * R_pow3.y + 6  * R_pow5.y * R.x);

        m0c0 = evaluation_coefficients.x * POLE32_MOMENT_X(m);
        m1c1 = evaluation_coefficients.y * POLE32_MOMENT_Y(m);
        m1c0 = evaluation_coefficients.x * POLE32_MOMENT_Y(m);
        m0c1 = evaluation_coefficients.y * POLE32_MOMENT_X(m);

        result.x -= (m0c0 + m1c1);
        result.y -= (m0c1 - m1c0);
$pp endif $

        // 64-pole
$pp if MULTIPOLE_ORDER >= 6 $
        r_pow2i_plus2_inv *= r2_inv;

        evaluation_coefficients.x =
          r_pow2i_plus2_inv * ( R_pow7.x + 35 * R_pow3.x * R_pow4.y - 21 * R_pow5.x * R_pow2.y - 7 * R_pow6.y * R.x);
        evaluation_coefficients.y =
          r_pow2i_plus2_inv * (-R_pow7.y - 35 * R_pow4.x * R_pow3.y + 21 * R_pow2.x * R_pow5.y + 7 * R_pow6.x * R.y);

        m0c0 = evaluation_coefficients.x * POLE64_MOMENT_X(m);
        m1c1 = evaluation_coefficients.y * POLE64_MOMENT_Y(m);
        m1c0 = evaluation_coefficients.x * POLE64_MOMENT_Y(m);
        m0c1 = evaluation_coefficients.y * POLE64_MOMENT_X(m);

        result.x -= (m0c0 + m1c1);
        result.y -= (m0c1 - m1c0);
$pp endif $

        return result;
      }


    )
  )
};


}

#endif

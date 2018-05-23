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

#ifndef INSTRUCTIONS_HPP
#define INSTRUCTIONS_HPP

#include <QCL/qcl_module.hpp>

#include "configuration.hpp"

namespace teralens {

QCL_STANDALONE_MODULE(instructions)
QCL_STANDALONE_SOURCE(
  QCL_IMPORT_CONSTANT(allow_fma_instructions)
  QCL_IMPORT_CONSTANT(allow_mad_instructions)
  R"(
    #if allow_fma_instructions==1
      #define FMA(a,b,c) fma(a,b,c)
    #else
      #define FMA(a,b,c) (a*b+c)
    #endif

    #if allow_mad_instructions==1
      #define MAD(a,b,c) mad(a,b,c)
    #else
      #define MAD(a,b,c) (a*b+c)
    #endif
  )"
)

}

#endif

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

#ifndef TERALENS_VERSION_HPP
#define TERALENS_VERSION_HPP

#include <boost/preprocessor/stringize.hpp>
#include <string>

#define TERALENS_VERSION_MAJOR 1
#define TERALENS_VERSION_MINOR 3
#define TERALENS_VERSION_PATCH 2
#define TERALENS_VERSION_ANNOTATION "Release"
#define TERALENS_GIT_BRANCH BOOST_PP_STRINGIZE(GIT_BRANCH)
#define TERALENS_GIT_COMMIT BOOST_PP_STRINGIZE(GIT_COMMIT_HASH)

namespace teralens {

static std::string version_string()
{
  return  std::to_string(TERALENS_VERSION_MAJOR)
     +"."+std::to_string(TERALENS_VERSION_MINOR)
     +"."+std::to_string(TERALENS_VERSION_PATCH)
     +"-"+std::string{TERALENS_GIT_BRANCH}
     +"/"+std::string{TERALENS_GIT_COMMIT};
}

static std::string annotated_version_string()
{
  return version_string()+" ("+std::string{TERALENS_VERSION_ANNOTATION}+")";
}



}

#endif

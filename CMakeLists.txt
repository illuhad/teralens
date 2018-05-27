
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wno-ignored-attributes -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=native -ffast-math -Wall -Wno-ignored-attributes -Wextra -Wno-unused-parameter")


cmake_minimum_required (VERSION 2.6)
project (teralens)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Choose the type of build, options are: Debug Release"
       FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCL REQUIRED)
find_package(Boost
  1.61      # Minimum version
  REQUIRED
  program_options)  # Fail with error if Boost is not found

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_definitions("-DGIT_COMMIT_HASH=${GIT_COMMIT_HASH}")
add_definitions("-DGIT_BRANCH=${GIT_BRANCH}")


include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIR}  ${OpenCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

add_executable(teralens_cpu src/teralens.cpp)
add_executable(teralens_gpu src/teralens.cpp)
add_executable(teralens_sampling_benchmark src/teralens_benchmark.cpp)

target_compile_definitions(teralens_cpu PRIVATE TERALENS_CPU_FALLBACK)

target_link_libraries (teralens_cpu ${OpenCL_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} -lcfitsio)
target_link_libraries (teralens_gpu ${OpenCL_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} -lcfitsio)
target_link_libraries (teralens_sampling_benchmark ${OpenCL_LIBRARIES} ${Boost_PROGRAM_OPTIONS_LIBRARY} -lcfitsio)





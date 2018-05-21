# Teralens - A parallel (quasar) microlensing code for multi-teraflop devices

Teralens is a modern implementation of a tree code for gravitational (quasar) microlensing, and fully parallelized for GPUs with OpenCL. A fallback CPU backend (which exploits the capabilities of multi-core CPUs) is available as well.
The general idea behind the microlensing tree code is described by Wambsganss (1999). While Teralens follows the same general principles, the actual algorithms used are dramatically different and tuned for maximum parallel efficiency.

## Computational method
Teralens contains two methods to calculate microlensing magnification patterns: A Barnes-Hut-like parallel tree code and a parallel brute-force method that directly sums up the deflection angle of the stars. The brute-force method is (by default) only used if the number of stars is smaller than 128, since then the latency to construct the tree is not always justified (but the exact crossover point depends on the hardware).

The tree code, the main method of Teralens, works as follows:
1. A tree is constructed in parallel on the GPU. The tree data structure is provided by my SpatialCL library (see http://github.com/illuhad/SpatialCL). This tree is a perfectly balanced binary tree constructed over the stars sorted along a Hilbert curve (note that this allows Teralens to efficiently process clustered star distributions as well!). While this tree choice can lead to an overlapping of the nodes (which may require more nodes to be accessed), it allows for a very fast tree construction.
   After a parallel sort of the stars on the GPU (along a Hilbert curve), the tree is constructed hierarchically from the lowest level up to the root. On each level, for each node, the node shape, center of mass and total mass (i.e. the monopole moment) is calculated in parallel from the (already calculated) child nodes. In a second processing step, the higher multipole moments (up to the 64-pole moment) are calculated for each level.
   To this end, a parallel reduce-by-key-like algorithm is executed on each level that adds the multipole moments due to each particle for each node.
2. A batch of primary ray positions on a grid is generated on the GPU.
3. For each primary ray position, the tree is queried in parallel. The goal of this step is 1.) to obtain a list of stars in the vicinity of the ray position that must be treated exactly, and 2.) to obtain interpolation coefficients for the long-range portion of the deflection angle (see step 4). Starting from the root, it is checked if the angular width of the node is smaller than a predefined opening angle (by default 0.5). If this is the case, is is assumed that the contribution of the particles within the tree node can be approximated by their multipole expansion. If so, the multipole expansion is evaluated at 25 interpolation points around the primary ray position.
   If the node cannot be approximated, its children must be considered. Note that the query order of the nodes in Teralens/SpatialCL is not the same as the straight-forward recursive depth-first search. Instead, it is a "relaxed" depth-first search where, when going up in the tree, the uncle node is directly accessed instead of first unwinding the callstack (plus, the implementation is purely iterative since this maps better to GPU hardware). This is possible because our tree is a pointer-free, perfectly balanced binary tree.
   If the algorithm has arrived at the lowest level in the tree (the stars), it adds the stars to the list of stars to be processed exactly for this ray. This list can hold (by default) up to 128 stars per ray. If this number should be exceeded, a wrong output is generated. This is however practically impossible since the number of exact stars is typically only around 20-30 per ray.
   When the query has finished, from the 25 interpolation points per ray the bicubic interpolation coefficients for 4 cells around the ray are calculated (Bicubic interpolation requires 16 evaluation points per cell, but interpolation points at the cell interfaces are shared such that 25 points suffice for the 4 cells).
4. Each interpolation cell is subsampled in a grid of 32x32 parallel rays (by default). At these locations, the long-range portion of the deflection is interpolated from the bicubic interpolation coefficients calculated previously, and the short-range portion is calculated by summing up the deflection due to the exact particles in the list for each point in the 32x32 grid. This means that, for each interpolation cell, 32x32=1024 secondary rays are obtained. Since we have 4 interpolation cells per primary ray, we obtain in total 4096 secondary rays from one primary ray (i.e. from one tree query).
5. From the deflection angles, the positions in the source plane are calculated and it is calculated which pixel is hit by the ray. The hit-counter for this pixel is then incremented (via an atomic integer addition).
Steps 2-5 are repeated until enough rays are obtained.

I would like to emphasize that, unlike earlier GPU-accelerated quasar microlensing codes (e.g. Thompson 2010), Teralens works _entirely_ on the GPU. The only objects transferred between CPU and GPU are the star sample in the beginning (copied to the GPU), and, in the end, the final magnification pattern (copied back to the CPU). In between these two steps, the calculation is fully carried out on the GPU. For future versions, it is planned to move the sampling of the stars to the GPU as well.

Teralens is the world's first quasar microlensing code with fully parallel implementations of all major stages of the computation (including the tree construction and binning of the rays in pixels). This allows it to achieve maximum parallel efficiency (if you do not have a background in parallel computing, this may be a good time to read up on Amdahl's law :).

## Credits
In general, you are free to use Teralens under the conditions of the GNU General Public License 3.0, but if you intend to use Teralens for scientific publications, I would like to kindly request the following:
* it is likely that a publication about Teralens will appear at some point in the future (this README will then be updated to point you to this publication). If this is the case, please cite this publication. Until then, please give credit to Aksel Alpay and provide a link to http://github.com/illuhad/teralens.
* if you have made any improvements to the code that may be of interest to the wider public (e.g. performance improvements, new features), please give something back and file a pull request for your improvements on github (or contact me directly).

## Hardware requirements
Teralens was written to be as non-restrictive as possible:
* You do _not_ need hardware from one specific vendor. Teralens should work with any GPU vendor (NVIDIA, AMD, in principle even Intel integrated GPUs). Unfortunately, at the moment I only have NVIDIA GPUs available for testing, but I strive for wide compatibility. If you have a GPU from a different vendor and encounter problems, it is considered a bug and I will do my best to fix it.
* You do _not_ need expensive data-center hardware (e.g. NVIDIA Tesla cards). Any off-the-shelf consumer grade GPU is fine. In fact, a Tesla card will give you no benefits compared to a consumer GPU that is from the same generation and has comparable clock speed, core count and memory bandwidth.
* Your GPU does _not_ need to have vast amounts of memory. Unless you want to create massive magnification patterns with more than 10 million stars, even 2GB should suffice.
* Teralens also runs decently on CPUs, if you do not have a GPU available at all. However, you will still need an OpenCL implementation from your CPU vendor (e.g. Intel). Additionally, most of my optimization efforts target GPUs. Within the same power envolope, Teralens will run significantly faster on a GPU.

## Software dependencies
For a successful compilation of Teralens, you need the following software:
* A C++11-compliant compiler
* `cmake`
* `cfitsio`
* An OpenCL implementation supporting at least OpenCL 1.2. Teralens is tested with the NVIDIA and Intel OpenCL implementations, but should also work on AMD hardware (but I do not have AMD GPUs available for testing at the moment -- reports on the experience of using Teralens on AMD GPUs or CPUs are highly appreciated!)
* The OpenCL C++ bindings. If you do not have them installed, you can get them here: https://github.com/KhronosGroup/OpenCL-CLHPP. In particular, the header file `cl2.hpp` is required.
* The `boost` C++ libraries. In particular, the `boost.compute` library. Please use the newest version available, since older versions may be affected by a memory leak. See https://github.com/boostorg/compute/issues/746.
* Teralens also depends on my SpatialCL and QuickCL libraries. See below how to fulfill these dependencies.

## Building

Teralens depends on a few other git repositories (QuickCL, SpatialCL). When cloning Teralens from the github repository, please make sure that you also obtain these libraries. This is achieved by executing the following command:
```
$ git clone --recurse-submodules https://github.com/illuhad/teralens.git
```

Then, create a build directory (can be anywhere on your system):
```
$ mkdir build
$ cd build
```
and configure and compile Teralens in your build directory
```
$ cmake <path-to-Teralens-directory>
$ make
```
Here, `<path-to-Teralens-directory>` should be replaced with the path to the Teralens directory (the one containing the `CMakeLists.txt` file)

If all went well, you should end up with several executables:
* `teralens_cpu` -- CPU version of Teralens
* `teralens_gpu` -- GPU version of Teralens
* `teralens_bench` -- Teralens benchmark, measures how fast your hardware can generate magnification patterns. Currently, this program requires a GPU.

You can test Teralens by just executing it without any command-line parameters. It will then create a magnification pattern named `teralens.fits` with a default set of parameters.
Use the `--help` option to see the available command-line parameters and the default parameter set. In general, it is usually not necessary to modify the Teralens code since all important aspects can be configured via command-line parameters.

## Troubleshooting
If you encounter problems, please do not hesitate to drop me an email: alpay at stud.uni-heidelberg.de

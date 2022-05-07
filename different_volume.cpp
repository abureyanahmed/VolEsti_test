

#include <boost/random.hpp>
#include "exact_vols.h"
#include "Eigen/Eigen"
#include <fstream>
#include "generators/known_polytope_generators.h"
#include "generators/z_polytopes_generators.h"
#include "misc/misc.h"
#include "misc/linear_extensions.h"
#include "random/uniform_int.hpp"
#include "random/normal_distribution.hpp"
#include "random/uniform_real_distribution.hpp"
#include "random_walks/random_walks.hpp"
#include "sampling/sampling.hpp"
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "volume/rotating.hpp"


#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_hpoly.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_gaussians.hpp"

int main()
{
    typedef double DBL;
    typedef Cartesian<DBL>    Kernel;
    typedef typename Kernel::Point    Pnt;
    typedef boost::mt19937    RanType;
    typedef HPolytope<Pnt> Hpltp;
    typedef VPolytope<Pnt> Vpltp;


    typedef BoostRandomNumberGenerator<boost::mt11213b, DBL> RNG;
    Hpltp HPoly = generate_cube<Hpltp>(10, false);
    BallWalk BW(3);

    // Volume estimation

    VPolytope<Pnt> VP2 = generate_cube<Vpltp>(2, true);

    double tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Cube-v cb = "
              << volume_cooling_balls<>(VP2).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;


    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball SOB = "
              << volume_sequence_of_balls<>(HPoly) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball CG = "
              << volume_cooling_gaussians<>(HPoly) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball CB = "
              << volume_cooling_balls<>(HPoly).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball SOB = "
              << volume_sequence_of_balls<CDHRWalk, RNG>(HPoly) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball CG = "
              << volume_cooling_gaussians<GaussianCDHRWalk, RNG>(HPoly) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball CB = "
              << volume_cooling_balls<CDHRWalk, RNG>(HPoly).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;


    typedef Zonotope<Pnt> Zonotope;
    Zonotope Z = gen_zonotope_uniform<Zonotope, RanType>(10, 15, 211);
    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Zono CB = "
              << volume_cooling_hpoly<CDHRWalk, RNG, Hpltp>(Z) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;


    return 0;

}

/*

Cube-v cb = 3.70939 , 0.573916
Ball SOB = 974.628 , 0.009225
Ball CG = 1143.6 , 0.012573
Ball CB = 1021.41 , 0.00346
Ball SOB = 998.957 , 0.008768
Ball CG = 1028.65 , 0.014736
Ball CB = 1037.81 , 0.003579
Zono CB = 6.24131e+20 , 0.959959

*/

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "Eigen/Eigen"
#include "exact_vols.h"
#include <fstream>
#include "generators/known_polytope_generators.h"
#include "random_walks/random_walks.hpp"
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"


int main()
{
    typedef double DBL;
    typedef Cartesian<DBL>    Kernel;
    typedef typename Kernel::Point    Pnt;
    typedef boost::mt19937    RandomType;
    typedef HPolytope<Pnt> Hpltp;
    typedef VPolytope<Pnt> Vpltp;

    std::cout << "Volume algorithm: Sequence of Balls" << std::endl << std::endl;

    Hpltp HP = generate_cube<Hpltp>(10, false);

    //Chebychev inner ball computation
    std::pair<Pnt,DBL> CheBall;
    CheBall = HP.ComputeInnerBall();

    // Initialization of parameters like, dimension, walk length etc
    int n = HP.dimension();
    int walk_len=10 + n/10;
    int n_threads=1;
    DBL e=1, err=0.1;
    double tstart;

    // Initialization for random number parameters
    int rnum = std::pow(e,-2) * 400 * n * std::log(n);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    RandomType rng(seed);
    boost::random::uniform_real_distribution<>(urdist);
    boost::random::uniform_real_distribution<> urdist1(-1,1);


    // We first estimate the volume of H-Polytopes

    std::cout << "Volume estimation on H-polytopes (cube-10)" << std::endl;

    typedef BoostRandomNumberGenerator<boost::mt11213b, DBL> RNG;

    // Consider different algorithms

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "BallWalk (cube) = "
              << volume_sequence_of_balls<BallWalk, RNG>(HP, e, walk_len) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    tstart = (double)clock()/(double)CLOCKS_PER_SEC;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "CDHRWalk (cube) = "
              << volume_sequence_of_balls<CDHRWalk, RNG>(HP, e, walk_len) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "RDHRWalk (cube) = "
              << volume_sequence_of_balls<RDHRWalk, RNG>(HP, e, walk_len) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "BilliardWalk (cube) = "
              << volume_sequence_of_balls<BilliardWalk, RNG>(HP, e, walk_len) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    std::cout << std::endl;

    // Volume estimation of V-Polytopes

    std::cout << "Volume estimation on V-polytopes (cross-10)" << std::endl;


    Vpltp VP;
    VP = generate_cross<Vpltp>(10, true);

    // Different algorithms

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Ball (cross) = "
              << volume_sequence_of_balls<BallWalk, RNG>(VP) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "RDHR (cross) = "
              << volume_sequence_of_balls<RDHRWalk, RNG>(VP) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "CDHR (cross) = "
              << volume_sequence_of_balls<CDHRWalk, RNG>(VP) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "Blrd (cross) = "
              << volume_sequence_of_balls<BilliardWalk, RNG>(VP) << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    return 0;
}

/*

Volume algorithm: Sequence of Balls

Volume estimation on H-polytopes (cube-10)
BallWalk (cube) = 975.278 , 0.118433
CDHRWalk (cube) = 1065.06 , 0.032248
RDHRWalk (cube) = 964.572 , 0.19565
BilliardWalk (cube) = 992.73 , 0.527455

Volume estimation on V-polytopes (cross-10)
Segmentation fault

*/

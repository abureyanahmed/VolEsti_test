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
    typedef boost::mt19937    RndmType;
    typedef HPolytope<Pnt> Hpltp;

    std::cout << "Volume algorithm: Cooling Balls" << std::endl << std::endl;

    Hpltp HP = generate_cube<Hpltp>(10, false);

    //Chebychev inner ball
    std::pair<Pnt,DBL> CheBall;
    CheBall = HP.ComputeInnerBall();

    // Parameters initialization
    int n = HP.dimension();
    int walk_len=10 + n/10;
    int n_threads=1;
    DBL e=1, err=0.1;
    DBL C=2.0,rt,frac=0.1;
    int N = 500 * ((int) C) + ((int) (n * n / 2));
    int W = 6*n*n+800;
    rt = 1.0-1.0/(DBL(n));

    int rnum = std::pow(e,-2) * 400 * n * std::log(n);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    RndmType rng(seed);
    boost::normal_distribution<> rdist(0,1);
    boost::random::uniform_real_distribution<>(urdist);
    boost::random::uniform_real_distribution<> urdist1(-1,1);

    double tstart;

    // Volume estimation on H-polytopes (cube-10)

    std::cout << "Volume estimation:" << std::endl;
    std::cout << "H-polytopes of cube-10:" << std::endl;

    // volume estimation

    typedef BoostRandomNumberGenerator<boost::mt11213b, DBL> RNG;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "BallWalk (cube) = "
              << volume_cooling_balls<BallWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;
    tstart = (double)clock()/(double)CLOCKS_PER_SEC;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "RDHRWalk (cube) = "
              << volume_cooling_balls<RDHRWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "CDHRWalk (cube) = "
              << volume_cooling_balls<CDHRWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    tstart = (double)clock()/(double)CLOCKS_PER_SEC;
    std::cout << "BirdWalk (cube) = "
              << volume_cooling_balls<BilliardWalk, RNG>(HP, e, walk_len).second << " , ";
    std::cout << (double)clock()/(double)CLOCKS_PER_SEC - tstart << std::endl;

    return 0;
}

/*

Volume algorithm: Cooling Balls

Volume estimation:
H-polytopes of cube-10:
BallWalk (cube) = 1013.51 , 0.008404
RDHRWalk (cube) = 1069.35 , 0.010959
CDHRWalk (cube) = 1103.62 , 0.003931
BirdWalk (cube) = 1099.14 , 0.033134

*/

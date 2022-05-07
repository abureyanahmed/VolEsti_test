#include "doctest.h"
#include <fstream>
#include <iostream>
#include "known_polytope_generators.h"
#include "misc.h"
#include "preprocess/min_sampling_covering_ellipsoid_rounding.hpp"
#include "preprocess/max_inscribed_ellipsoid_rounding.hpp"
#include "preprocess/svd_rounding.hpp"
#include "random.hpp"
#include "random/uniform_int.hpp"
#include "random/normal_distribution.hpp"
#include "random/uniform_real_distribution.hpp"
#include "random_walks/random_walks.hpp"
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"

template <typename NT>
NT factorial(NT n)
{
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

template <typename NT>
void test_values(NT volume, NT expected, NT exact)
{
    std::cout << "Computed volume " << volume << std::endl;
    std::cout << "Expected volume = " << expected << std::endl;
    std::cout << "Relative error (expected) = "
              << std::abs((volume-expected)/expected) << std::endl;
    std::cout << "Relative error (exact) = "
              << std::abs((volume-exact)/exact) << std::endl;
    CHECK((std::abs((volume - exact)/exact) < 0.2 || 
            std::abs((volume - expected)/expected) < 0.00001));
}

template <class Polytope>
void rounding_min_ellipsoid_test(Polytope &HP,
                                 double const& expectedBall,
                                 double const& expectedCDHR,
                                 double const& expectedRDHR,
                                 double const& expectedBilliard,
                                 double const& exact)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename Polytope::MT MT;
    typedef typename Polytope::VT VT;

    int d = HP.dimension();

    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 5> RNGType;
    RNGType rng(d);

    std::pair<Point, NT> InnerBall = HP.ComputeInnerBall();
    std::tuple<MT, VT, NT> res = min_sampling_covering_ellipsoid_rounding<CDHRWalk, MT, VT>(HP, InnerBall,
                                                                                            10 + 10 * d, rng);

    // Setup the parameters
    int walk_len = 1;
    NT e = 0.1;

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;


    //TODO: low accuracy in high dimensions
    //NT volume = res.second * volume_cooling_balls<BallWalk, RNGType>(HP, e, walk_len);
    //test_values(volume, expectedBall, exact);

    NT volume = std::get<2>(res) * volume_cooling_balls<CDHRWalk, RNGType>(HP, e, walk_len).second;
    test_values(volume, expectedCDHR, exact);

    volume = std::get<2>(res) * volume_cooling_balls<RDHRWalk, RNGType>(HP, e, 2*walk_len).second;
    test_values(volume, expectedRDHR, exact);

    volume = std::get<2>(res) * volume_cooling_balls<BilliardWalk, RNGType>(HP, e, walk_len).second;
    test_values(volume, expectedBilliard, exact);
}


template <class Polytope>
void rounding_max_ellipsoid_test(Polytope &HP,
                                 double const& expectedBall,
                                 double const& expectedCDHR,
                                 double const& expectedRDHR,
                                 double const& expectedBilliard,
                                 double const& exact)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename Polytope::MT MT;
    typedef typename Polytope::VT VT;

    int d = HP.dimension();

    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 5> RNGType;
    RNGType rng(d);

    std::pair<Point, NT> InnerBall = HP.ComputeInnerBall();
    std::tuple<MT, VT, NT> res = max_inscribed_ellipsoid_rounding<MT, VT, NT>(HP, InnerBall.first);

    // Setup the parameters
    int walk_len = 1;
    NT e = 0.1;

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;

    NT volume = std::get<2>(res) * volume_cooling_balls<BilliardWalk, RNGType>(HP, e, walk_len).second;
    test_values(volume, expectedBilliard, exact);
}


template <class Polytope>
void rounding_svd_test(Polytope &HP,
                       double const& expectedBall,
                       double const& expectedCDHR,
                       double const& expectedRDHR,
                       double const& expectedBilliard,
                       double const& exact)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;
    typedef typename Polytope::MT MT;
    typedef typename Polytope::VT VT;

    int d = HP.dimension();

    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 5> RNGType;
    RNGType rng(d);

    std::pair<Point, NT> InnerBall = HP.ComputeInnerBall();
    std::tuple<MT, VT, NT> res = svd_rounding<CDHRWalk, MT, VT>(HP, InnerBall, 10 + 10 * d, rng);

    // Setup the parameters
    int walk_len = 1;
    NT e = 0.1;

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;

    NT volume = std::get<2>(res) * volume_cooling_balls<BilliardWalk, RNGType>(HP, e, walk_len).second;
    test_values(volume, expectedBilliard, exact);
}


template <typename NT>
void call_test_min_ellipsoid() {
    typedef Cartesian <NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef HPolytope <Point> Hpolytope;
    Hpolytope P;

    std::cout << "\n--- Testing rounding of H-skinny_cube5" << std::endl;
    P = generate_skinny_cube<Hpolytope>(5);
    rounding_min_ellipsoid_test(P, 0, 3070.64, 3188.25, 3140.6, 3200.0);

    std::cout << "\n--- Testing rounding of H-skinny_cube10" << std::endl;

    P = generate_skinny_cube<Hpolytope>(10);
    rounding_min_ellipsoid_test(P, 0, 122550, 108426, 105003.0, 102400.0);
}


template <typename NT>
void call_test_max_ellipsoid() {
    typedef Cartesian <NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef HPolytope <Point> Hpolytope;
    Hpolytope P;

    std::cout << "\n--- Testing rounding of H-skinny_cube5" << std::endl;
    P = generate_skinny_cube<Hpolytope>(5);
    rounding_max_ellipsoid_test(P, 0, 3070.64, 3188.25, 3140.6, 3200.0);
}


template <typename NT>
void call_test_svd() {
    typedef Cartesian <NT> Kernel;
    typedef typename Kernel::Point Point;
    typedef HPolytope <Point> Hpolytope;
    Hpolytope P;

    std::cout << "\n--- Testing rounding of H-skinny_cube5" << std::endl;
    P = generate_skinny_cube<Hpolytope>(5);
    rounding_svd_test(P, 0, 3070.64, 3188.25, 3140.6, 3200.0);
}


TEST_CASE("round_min_ellipsoid") {
    call_test_min_ellipsoid<double>();
}

TEST_CASE("round_max_ellipsoid") {
    call_test_max_ellipsoid<double>();
}

TEST_CASE("round_svd") {
    call_test_svd<double>();
}

/*

[doctest] doctest version is "1.2.9"
[doctest] run with "--help" for options

--- Testing rounding of H-skinny_cube5
Number type: d
Computed volume 3071.67
Expected volume = 3070.64
Relative error (expected) = 0.000336825
Relative error (exact) = 0.0401018
Computed volume 3129.59
Expected volume = 3188.25
Relative error (expected) = 0.0183995
Relative error (exact) = 0.0220038
Computed volume 3163.97
Expected volume = 3140.6
Relative error (expected) = 0.00744021
Relative error (exact) = 0.0112604

--- Testing rounding of H-skinny_cube10
Number type: d
Computed volume 102149
Expected volume = 122550
Relative error (expected) = 0.166472
Relative error (exact) = 0.00245223
Computed volume 94487.6
Expected volume = 108426
Relative error (expected) = 0.128552
Relative error (exact) = 0.0772695
Computed volume 97672.9
Expected volume = 105003
Relative error (expected) = 0.0698087
Relative error (exact) = 0.0461633

--- Testing rounding of H-skinny_cube5
Number type: d
Computed volume 3262.77
Expected volume = 3140.6
Relative error (expected) = 0.0388998
Relative error (exact) = 0.0196153

--- Testing rounding of H-skinny_cube5
Number type: d
Computed volume 3160.01
Expected volume = 3140.6
Relative error (expected) = 0.00617984
Relative error (exact) = 0.0124974
===============================================================================
[doctest] test cases:      3 |      3 passed |      0 failed |      0 skipped
[doctest] assertions:      8 |      8 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

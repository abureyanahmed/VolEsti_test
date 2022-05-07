#include "doctest.h"
#include <fstream>
#include <iostream>
#include "known_polytope_generators.h"
#include "misc.h"
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
    CHECK(std::abs((volume - exact)/exact) < 0.2);
}

template <class Polytope>
void test_volume(Polytope &P,
                 double const& expectedBall,
                 double const& expectedCDHR,
                 double const& expectedRDHR,
                 double const& expectedBilliard,
                 double const& exact)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;

    // Setup the parameters
    int walk_len = 10 + P.dimension()/10;
    NT e=0.1;

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 3> RNGType;

    //TODO: low accuracy in high dimensions
    Polytope P1(P.dimension(), P.get_mat(), P.get_vec());
    NT volume = volume_cooling_balls<BallWalk, RNGType>(P1, e, walk_len).second;
    test_values(volume, expectedBall, exact);

    Polytope P2(P.dimension(), P.get_mat(), P.get_vec());
    volume = volume_cooling_balls<CDHRWalk, RNGType>(P2, e, walk_len).second;
    test_values(volume, expectedCDHR, exact);

    Polytope P3(P.dimension(), P.get_mat(), P.get_vec());
    volume = volume_cooling_balls<RDHRWalk, RNGType>(P3, e, walk_len).second;
    test_values(volume, expectedRDHR, exact);

    Polytope P4(P.dimension(), P.get_mat(), P.get_vec());
    volume = volume_cooling_balls<BilliardWalk, RNGType>(P4, e, walk_len).second;
    test_values(volume, expectedBilliard, exact);
}

template <typename NT>
void call_test_cube(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef boost::mt19937 RNGType;
    typedef VPolytope<Point> Vpolytope;


    std::cout << "--- Testing volume of V-cube2" << std::endl;
    Vpolytope P1 = generate_cube<Vpolytope>(2, true);
    test_volume(P1, 4.43443, 4.129, 4.43443, 4.40191, 4);

    std::cout << "--- Testing volume of V-cube5" << std::endl;
    Vpolytope P2 = generate_cube<Vpolytope>(5, true);
    test_volume(P2, 32, 32, 32, 32, 32);


}

template <typename NT>
void call_test_cube_float(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef boost::mt19937 RNGType;
    typedef VPolytope<Point> Vpolytope;
    Vpolytope P;

    std::cout << "--- Testing volume of V-cube10 (float)" << std::endl;
    P = generate_cube<Vpolytope>(10, true);
    test_volume(P, 1000.55, 1024, 1024, 1024, 1024);

    std::cout << "--- Testing volume of V-cube20 (float)" << std::endl;
    P = generate_cube<Vpolytope>(20, true);
    test_volume(P, 1114192.7854272256,
                1048576,
                1048576,
                1048576,
                1048576);
}

template <typename NT>
void call_test_cross(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef boost::mt19937 RNGType;
    typedef VPolytope<Point> Vpolytope;
    Vpolytope P;

    std::cout << "--- Testing volume of V-cross5" << std::endl;
    P = generate_cross<Vpolytope>(5, true);
    test_volume(P,
                0.28425,
                0.273255,
                0.28413,
                0.286071,
                0.266666667);

    std::cout << "--- Testing volume of V-cross10" << std::endl;
    P = generate_cross<Vpolytope>(10, true);
    test_volume(P,
                0.000283841,
                0.00031188,
                0.000284841,
                0.00027759,
                0.0002821869);

    std::cout << "--- Testing volume of V-cross20" << std::endl;
    P = generate_cross<Vpolytope>(20, true);
    test_volume(P,
                4.16807 * std::pow(10,-13),
                4.42692 * std::pow(10,-13),
                4.19453 * std::pow(10,-13),
                4.63423 * std::pow(10,-13),
                std::pow(2.0,20.0) / factorial(20.0));
}

template <typename NT>
void call_test_simplex() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;

    typedef boost::mt19937 RNGType;
    typedef VPolytope<Point> Vpolytope;
    Vpolytope P;

    std::cout << "--- Testing volume of V-simplex5" << std::endl;
    P = generate_simplex<Vpolytope>(5, true);
    test_volume(P,
                0.00846587,
                0.0096107,
                0.00842591,
                0.00855401,
                1.0 / factorial(5.0));

    std::cout << "--- Testing volume of V-simplex10" << std::endl;
    P = generate_simplex<Vpolytope>(10, true);
    test_volume(P,
                2.35669 * std::pow(10,-7),
                3.00778 * std::pow(10,-7),
                3.0366 * std::pow(10,-7),
                2.72952 * std::pow(10,-7),
                1.0 / factorial(10.0));
/* too slow
    std::cout << "--- Testing volume of V-simplex20" << std::endl;
    P = gen_simplex<Vpolytope>(20, true);
    test_volume(P,
                1.13981 * std::pow(10,-19),
                3.63355 * std::pow(10,-19),
                4.46469 * std::pow(10,-19),
                4.22932 * std::pow(10,-19),
                1.0 / factorial(20.0));
*/
}


TEST_CASE("cube") {
    //TODO: Runtime error, check ComputeInnerBall()
    call_test_cube<double>();
    //call_test_cube_float<float>();
}

TEST_CASE("cross") {
    call_test_cross<double>();
}

TEST_CASE("simplex") {
    call_test_simplex<double>();
}

/*

[doctest] doctest version is "1.2.9"
[doctest] run with "--help" for options
--- Testing volume of V-cube2
Number type: d
Computed volume 4.4286
Expected volume = 4.43443
Relative error (expected) = 0.00131534
Relative error (exact) = 0.107149
Computed volume 4.02131
Expected volume = 4.129
Relative error (expected) = 0.0260807
Relative error (exact) = 0.00532822
Computed volume 4.41407
Expected volume = 4.43443
Relative error (expected) = 0.00459056
Relative error (exact) = 0.103518
Computed volume 4.32045
Expected volume = 4.40191
Relative error (expected) = 0.0185046
Relative error (exact) = 0.0801136
--- Testing volume of V-cube5
Number type: d
Computed volume 35.8128
Expected volume = 32
Relative error (expected) = 0.119149
Relative error (exact) = 0.119149
Computed volume 33.2377
Expected volume = 32
Relative error (expected) = 0.0386784
Relative error (exact) = 0.0386784
Computed volume 35.8286
Expected volume = 32
Relative error (expected) = 0.119643
Relative error (exact) = 0.119643
Computed volume 35.4056
Expected volume = 32
Relative error (expected) = 0.106425
Relative error (exact) = 0.106425
--- Testing volume of V-cross5
Number type: d
Computed volume 0.274799
Expected volume = 0.28425
Relative error (expected) = 0.0332491
Relative error (exact) = 0.030496
Computed volume 0.273051
Expected volume = 0.273255
Relative error (expected) = 0.000746824
Relative error (exact) = 0.023941
Computed volume 0.27562
Expected volume = 0.28413
Relative error (expected) = 0.0299524
Relative error (exact) = 0.0335736
Computed volume 0.278685
Expected volume = 0.286071
Relative error (expected) = 0.0258199
Relative error (exact) = 0.0450676
--- Testing volume of V-cross10
Number type: d
Computed volume 0.000306295
Expected volume = 0.000283841
Relative error (expected) = 0.0791059
Relative error (exact) = 0.0854313
Computed volume 0.000289388
Expected volume = 0.00031188
Relative error (expected) = 0.0721187
Relative error (exact) = 0.0255176
Computed volume 0.000307906
Expected volume = 0.000284841
Relative error (expected) = 0.0809734
Relative error (exact) = 0.0911405
Computed volume 0.000291618
Expected volume = 0.00027759
Relative error (expected) = 0.050534
Relative error (exact) = 0.0334205
--- Testing volume of V-cross20
Number type: d
Computed volume 3.99425e-13
Expected volume = 4.16807e-13
Relative error (expected) = 0.0417019
Relative error (exact) = 0.0732548
Computed volume 4.45855e-13
Expected volume = 4.42692e-13
Relative error (expected) = 0.00714419
Relative error (exact) = 0.0344703
Computed volume 3.90936e-13
Expected volume = 4.19453e-13
Relative error (expected) = 0.0679866
Relative error (exact) = 0.0929523
Computed volume 3.9665e-13
Expected volume = 4.63423e-13
Relative error (expected) = 0.144087
Relative error (exact) = 0.0796951
--- Testing volume of V-simplex5
Number type: d
Computed volume 0.00862129
Expected volume = 0.00846587
Relative error (expected) = 0.0183585
Relative error (exact) = 0.0345548
Computed volume 0.00928001
Expected volume = 0.0096107
Relative error (expected) = 0.0344087
Relative error (exact) = 0.113601
Computed volume 0.0088914
Expected volume = 0.00842591
Relative error (expected) = 0.0552455
Relative error (exact) = 0.0669684
Computed volume 0.00860877
Expected volume = 0.00855401
Relative error (expected) = 0.00640204
Relative error (exact) = 0.0330528
--- Testing volume of V-simplex10
Number type: d
Computed volume 2.32759e-07
Expected volume = 2.35669e-07
Relative error (expected) = 0.0123466
Relative error (exact) = 0.155363
Computed volume 2.82581e-07
Expected volume = 3.00778e-07
Relative error (expected) = 0.0604982
Relative error (exact) = 0.0254317
Computed volume 2.54152e-07
Expected volume = 3.0366e-07
Relative error (expected) = 0.163038
Relative error (exact) = 0.0777337
Computed volume 2.50521e-07
Expected volume = 2.72952e-07
Relative error (expected) = 0.0821804
Relative error (exact) = 0.0909105
===============================================================================
[doctest] test cases:      3 |      3 passed |      0 failed |      0 skipped
[doctest] assertions:     28 |     28 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

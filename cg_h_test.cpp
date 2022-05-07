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
    CHECK((std::abs((volume - exact)/exact) < 0.2 || 
           std::abs((volume - expected)/expected) < 0.00001));
}

template <class Polytope>
void test_volume(Polytope &HP,
                 double const& expectedBall,
                 double const& expectedCDHR,
                 double const& expectedRDHR,
                 double const& exact,
                 bool birk = false)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;

    // Setup the parameters
    int walk_len = 10 + HP.dimension()/10;
    NT e=0.1;
    NT volume;

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 3> RNGType;

    // TODO: low accuracy in high-dimensions
    if (!birk) {
        volume = volume_cooling_gaussians<GaussianBallWalk, RNGType>(HP, e, walk_len);
        test_values(volume, expectedBall, exact);
    }

    volume = volume_cooling_gaussians<GaussianCDHRWalk, RNGType>(HP, e, walk_len);
    test_values(volume, expectedCDHR, exact);

    volume = volume_cooling_gaussians<GaussianRDHRWalk, RNGType>(HP, e, walk_len);
    test_values(volume, expectedRDHR, exact);
}

template <typename NT>
void call_test_cube(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;

    std::cout << "--- Testing volume of H-cube10" << std::endl;
    P = generate_cube<Hpolytope>(10, false);
    test_volume(P, 1079.56, 1110.92, 1113.93, 1024);

    std::cout << "--- Testing volume of H-cube20" << std::endl;
    P = generate_cube<Hpolytope>(20, false);
    test_volume(P, 1.1025e+06, 1.05174e+06, 995224, 1048576);
}

template <typename NT>
void call_test_cube_float(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;
/*
    std::cout << "--- Testing volume of H-cube10" << std::endl;
    P = generate_cube<Hpolytope>(10, false);
    test_volume(P, 1000.55, 1024);

    std::cout << "--- Testing volume of H-cube20" << std::endl;
    P = generate_cube<Hpolytope>(20, false);
    test_volume(P, 1114192.7854272256, 1048576);
    */
}

template <typename NT>
void call_test_cross(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef HPolytope<Point> Hpolytope;

    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 123> RNGType;

    std::cout << "--- Testing volume of H-cross10" << std::endl;
    Hpolytope P = generate_cross<Hpolytope>(10, false);
    test_volume(P,
                0.000292199,
                0.000274014,
                0.000294463,
                0.0002821869);
}

template <typename NT>
void call_test_birk() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;

    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 123> RNGType;

    std::cout << "--- Testing volume of H-birk3" << std::endl;
    P = generate_birkhoff<Hpolytope>(3);
    test_volume(P, 0.116678, 0.122104, 0.11326, 0.125, true);

    std::cout << "--- Testing volume of H-birk4" << std::endl;
    P = generate_birkhoff<Hpolytope>(4);
    test_volume(P,
                0.000450761,
                0.00108943,
                0.00110742,
                0.000970018,
                true);

    std::cout << "--- Testing volume of H-birk5" << std::endl;
    P = generate_birkhoff<Hpolytope>(5);
    test_volume(P,
                2.97522e-08,
                2.00743e-07,
                2.05779e-07,
                2.25  * std::pow(10,-7),
                true);

    std::cout << "--- Testing volume of H-birk6" << std::endl;
    P = generate_birkhoff<Hpolytope>(6);
    test_volume(P,
                3.66375e-19,
                7.51051 * std::pow(10,-13),
                8.20587e-13,
                9.455459196 * std::pow(10,-13),
                true);
}

template <typename NT>
void call_test_prod_simplex() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;

    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;

    std::cout << "--- Testing volume of H-prod_simplex5" << std::endl;
    P = generate_prod_simplex<Hpolytope>(5);
    test_volume(P,
                6.3448 * std::pow(10,-5),
                6.94695 * std::pow(10,-5),
                6.13242e-05,
                std::pow(1.0 / factorial(5.0), 2));

    std::cout << "--- Testing volume of H-prod_simplex10" << std::endl;
    P = generate_prod_simplex<Hpolytope>(10);
    test_volume(P,
                1.66017 * std::pow(10,-14),
                8.48116 * std::pow(10,-14),
                6.90898e-14,
                std::pow(1.0 / factorial(10.0), 2));

    std::cout << "--- Testing volume of H-prod_simplex15" << std::endl;
    P = generate_prod_simplex<Hpolytope>(15);
    test_volume(P,
                2.0232 * std::pow(10,-29),
                5.4624 * std::pow(10,-25),
                6.95082e-25,
                std::pow(1.0 / factorial(15.0), 2));
}

template <typename NT>
void call_test_simplex() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;

    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;

    std::cout << "--- Testing volume of H-simplex10" << std::endl;
    P = generate_simplex<Hpolytope>(10, false);
    test_volume(P,
                2.14048 * std::pow(10,-7),
                2.70598 * std::pow(10,-7),
                2.53893e-07,
                1.0 / factorial(10.0));

    std::cout << "--- Testing volume of H-simplex20" << std::endl;
    P = generate_simplex<Hpolytope>(20, false);
    test_volume(P,
                2.00646 * std::pow(10,-21),
                4.16845 * std::pow(10,-19),
                3.79918e-19,
                1.0 / factorial(20.0));

    std::cout << "--- Testing volume of H-simplex30" << std::endl;
    P = generate_simplex<Hpolytope>(30, false);
    test_volume(P,
                2.31348 * std::pow(10,-35),
                4.02288 * std::pow(10,-33),
                3.47743e-33,
                1.0 / factorial(30.0));
}

template <typename NT>
void call_test_skinny_cube() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 123> RNGType;

    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;

    std::cout << "--- Testing volume of H-skinny_cube10" << std::endl;
    //TODO: needs rounding
    //P = gen_skinny_cube<Hpolytope>(10);
    //test_volume(P, 15591.1, 102400.0);

    //std::cout << "--- Testing volume of H-skinny_cube20" << std::endl;
    //P = gen_skinny_cube<Hpolytope>(20);
    //test_volume(P, 104857600, 104857600.0);
}


TEST_CASE("cube") {
    call_test_cube<double>();
    call_test_cube_float<float>();
}

TEST_CASE("cross") {
    call_test_cross<double>();
}

TEST_CASE("birk") {
    call_test_birk<double>();
}

TEST_CASE("prod_simplex") {
    call_test_prod_simplex<double>();
}

TEST_CASE("simplex") {
    call_test_simplex<double>();
}

TEST_CASE("skinny_cube") {
    call_test_skinny_cube<double>();
}

/*

[doctest] doctest version is "1.2.9"
[doctest] run with "--help" for options
--- Testing volume of H-cube10
Number type: d
Computed volume 1043.28
Expected volume = 1079.56
Relative error (expected) = 0.0336059
Relative error (exact) = 0.0188285
Computed volume 1014.29
Expected volume = 1110.92
Relative error (expected) = 0.0869816
Relative error (exact) = 0.00948205
Computed volume 1010.16
Expected volume = 1113.93
Relative error (expected) = 0.0931585
Relative error (exact) = 0.0135176
--- Testing volume of H-cube20
Number type: d
Computed volume 1.04892e+06
Expected volume = 1.1025e+06
Relative error (expected) = 0.048602
Relative error (exact) = 0.000324533
Computed volume 1.07887e+06
Expected volume = 1.05174e+06
Relative error (expected) = 0.0257921
Relative error (exact) = 0.0288874
Computed volume 1.00363e+06
Expected volume = 995224
Relative error (expected) = 0.00844307
Relative error (exact) = 0.0428669
--- Testing volume of H-cross10
Number type: d
Computed volume 0.000295308
Expected volume = 0.000292199
Relative error (expected) = 0.0106403
Relative error (exact) = 0.0464982
Computed volume 0.000285244
Expected volume = 0.000274014
Relative error (expected) = 0.0409848
Relative error (exact) = 0.0108351
Computed volume 0.000270528
Expected volume = 0.000294463
Relative error (expected) = 0.0812826
Relative error (exact) = 0.0413152
--- Testing volume of H-birk3
Number type: d
Computed volume 0.127629
Expected volume = 0.122104
Relative error (expected) = 0.0452455
Relative error (exact) = 0.0210292
Computed volume 0.121377
Expected volume = 0.11326
Relative error (expected) = 0.0716708
Relative error (exact) = 0.0289805
--- Testing volume of H-birk4
Number type: d
Computed volume 0.00110265
Expected volume = 0.00108943
Relative error (expected) = 0.0121379
Relative error (exact) = 0.136735
Computed volume 0.000986723
Expected volume = 0.00110742
Relative error (expected) = 0.108989
Relative error (exact) = 0.0172216
--- Testing volume of H-birk5
Number type: d
Computed volume 2.00331e-07
Expected volume = 2.00743e-07
Relative error (expected) = 0.0020504
Relative error (exact) = 0.109638
Computed volume 1.99681e-07
Expected volume = 2.05779e-07
Relative error (expected) = 0.0296348
Relative error (exact) = 0.11253
--- Testing volume of H-birk6
Number type: d
Computed volume 7.51051e-13
Expected volume = 7.51051e-13
Relative error (expected) = 1.96852e-07
Relative error (exact) = 0.205696
Computed volume 8.16977e-13
Expected volume = 8.20587e-13
Relative error (expected) = 0.00439873
Relative error (exact) = 0.135973
--- Testing volume of H-prod_simplex5
Number type: d
Computed volume 6.21536e-05
Expected volume = 6.3448e-05
Relative error (expected) = 0.0204003
Relative error (exact) = 0.104988
Computed volume 7.0829e-05
Expected volume = 6.94695e-05
Relative error (expected) = 0.01957
Relative error (exact) = 0.0199379
Computed volume 7.01432e-05
Expected volume = 6.13242e-05
Relative error (expected) = 0.143809
Relative error (exact) = 0.0100616
--- Testing volume of H-prod_simplex10
Number type: d
Computed volume 1.66017e-14
Expected volume = 1.66017e-14
Relative error (expected) = 2.07755e-07
Relative error (exact) = 0.781386
Computed volume 8.23029e-14
Expected volume = 8.48116e-14
Relative error (expected) = 0.0295799
Relative error (exact) = 0.08378
Computed volume 7.23421e-14
Expected volume = 6.90898e-14
Relative error (expected) = 0.0470735
Relative error (exact) = 0.0473856
--- Testing volume of H-prod_simplex15
Number type: d
Computed volume 2.0232e-29
Expected volume = 2.0232e-29
Relative error (expected) = 1.98012e-06
Relative error (exact) = 0.999965
Computed volume 6.07639e-25
Expected volume = 5.4624e-25
Relative error (expected) = 0.112403
Relative error (exact) = 0.0390701
Computed volume 5.40931e-25
Expected volume = 6.95082e-25
Relative error (expected) = 0.221774
Relative error (exact) = 0.075001
--- Testing volume of H-simplex10
Number type: d
Computed volume 2.14048e-07
Expected volume = 2.14048e-07
Relative error (expected) = 1.50419e-06
Relative error (exact) = 0.223264
Computed volume 2.58542e-07
Expected volume = 2.70598e-07
Relative error (expected) = 0.0445548
Relative error (exact) = 0.0618044
Computed volume 2.5932e-07
Expected volume = 2.53893e-07
Relative error (expected) = 0.0213752
Relative error (exact) = 0.0589796
--- Testing volume of H-simplex20
Number type: d
Computed volume 2.00646e-21
Expected volume = 2.00646e-21
Relative error (expected) = 3.06366e-07
Relative error (exact) = 0.995118
Computed volume 4.57044e-19
Expected volume = 4.16845e-19
Relative error (expected) = 0.0964373
Relative error (exact) = 0.111944
Computed volume 4.28323e-19
Expected volume = 3.79918e-19
Relative error (expected) = 0.12741
Relative error (exact) = 0.0420687
--- Testing volume of H-simplex30
Number type: d
Computed volume 2.31348e-35
Expected volume = 2.31348e-35
Relative error (expected) = 1.90562e-06
Relative error (exact) = 0.993863
Computed volume 4.0693e-33
Expected volume = 4.02288e-33
Relative error (expected) = 0.0115387
Relative error (exact) = 0.0793931
Computed volume 3.93922e-33
Expected volume = 3.47743e-33
Relative error (expected) = 0.132797
Relative error (exact) = 0.04489
--- Testing volume of H-skinny_cube10
===============================================================================
[doctest] test cases:      6 |      6 passed |      0 failed |      0 skipped
[doctest] assertions:     35 |     35 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

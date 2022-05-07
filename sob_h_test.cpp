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
                 double const& expectedBilliard,
                 double const& exact)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;

    // Setup the parameters
    int walk_len = 10 + HP.dimension()/10;
    NT e=1;

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 3> RNGType;

    //TODO: low accuracy in high dimensions
    NT volume = volume_sequence_of_balls<BallWalk, RNGType>(HP, e, walk_len);
    test_values(volume, expectedBall, exact);

    volume = volume_sequence_of_balls<CDHRWalk, RNGType>(HP, e, walk_len);
    test_values(volume, expectedCDHR, exact);

    volume = volume_sequence_of_balls<RDHRWalk, RNGType>(HP, e, walk_len);
    test_values(volume, expectedRDHR, exact);

    //TODO: slow and with low accuracy
    //volume = volume_sequence_of_balls<BilliardWalk, RNGType>(HP, e, walk_len);
    test_values(volume, expectedBilliard, exact);
}


template <typename NT>
void call_test_cube(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;

    std::cout << "--- Testing volume of H-cube10" << std::endl;
    P = generate_cube<Hpolytope>(10, false);
    test_volume(P, 1014.69, 1049.22, 1055.73, 1055.73, 1024);

    std::cout << "--- Testing volume of H-cube20" << std::endl;
    P = std::move(generate_cube<Hpolytope>(20, false));
    test_volume(P, 1.02978e+06, 1056180, 1058830, 1058830, 1048576);
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
    test_volume(P, 1000.55, 1024, 1024, 1024, 1024);

    std::cout << "--- Testing volume of H-cube20" << std::endl;
    P = generate_cube<Hpolytope>(20, false);
    test_volume(P,
                1114192.7854272256,
                1048576,
                1048576,
                1048576,
                1048576);
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
                0.000283788,
                0.000280815,
                0.000296745,
                0.000296745,
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
    test_volume(P,
                0.130806,
                0.126776,
                0.122177,
                0.122177,
                0.125);

    std::cout << "--- Testing volume of H-birk4" << std::endl;
    P = generate_birkhoff<Hpolytope>(4);
    test_volume(P,
                0.00112925,
                0.000898527,
                0.000945447,
                0.000945447,
                0.000970018);

    std::cout << "--- Testing volume of H-birk5" << std::endl;
    P = generate_birkhoff<Hpolytope>(5);
    test_volume(P,
                1.75176 * std::pow(10,-7),
                2.07943 * std::pow(10,-7),
                2.80779e-07,
                2.80779e-07,
                0.000000225);

    std::cout << "--- Testing volume of H-birk6" << std::endl;
    P = generate_birkhoff<Hpolytope>(6);
    test_volume(P,
                5.71076 * std::pow(10,-13),
                9.48912 * std::pow(10,-13),
                6.96476e-13,
                6.96476e-13,
                0.0000000000009455459196);
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
                7.35223 * std::pow(10,-5),
                6.86576 * std::pow(10,-5),
                7.43136 * std::pow(10,-5),
                7.43136 * std::pow(10,-5),
                std::pow(1.0 / factorial(5.0), 2));

    std::cout << "--- Testing volume of H-prod_simplex10" << std::endl;
    P = generate_prod_simplex<Hpolytope>(10);
    test_volume(P,
                7.38925 * std::pow(10,-14),
                8.01351 * std::pow(10,-14),
                8.27387 * std::pow(10,-14),
                8.27387 * std::pow(10,-14),
                std::pow(1.0 / factorial(10.0), 2));

    std::cout << "--- Testing volume of H-prod_simplex15" << std::endl;
    P = generate_prod_simplex<Hpolytope>(15);
    test_volume(P,
                5.61238 * std::pow(10,-25),
                5.87558 * std::pow(10,-25),
                5.48179 * std::pow(10,-25),
                5.48179 * std::pow(10,-25),
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
                2.98074 * std::pow(10,-7),
                2.52756 * std::pow(10,-7),
                2.89366 * std::pow(10,-7),
                2.89366 * std::pow(10,-7),
                1.0 / factorial(10.0));

    std::cout << "--- Testing volume of H-simplex20" << std::endl;
    P = generate_simplex<Hpolytope>(20, false);
    test_volume(P,
                4.64611 * std::pow(10,-19),
                4.4317 * std::pow(10,-19),
                4.16737 * std::pow(10,-19),
                4.16737 * std::pow(10,-19),
                1.0 / factorial(20.0));

    std::cout << "--- Testing volume of H-simplex30" << std::endl;
    P = generate_simplex<Hpolytope>(30, false);
    test_volume(P,
                3.65853 * std::pow(10,-33),
                3.86474 * std::pow(10,-33),
                4.04136 * std::pow(10,-33),
                4.04136 * std::pow(10,-33),
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
Computed volume 985.088
Expected volume = 1014.69
Relative error (expected) = 0.0291733
Relative error (exact) = 0.0379999
Computed volume 1049.22
Expected volume = 1049.22
Relative error (expected) = 3.55462e-07
Relative error (exact) = 0.0246293
Computed volume 969.683
Expected volume = 1055.73
Relative error (expected) = 0.0815047
Relative error (exact) = 0.0530439
Computed volume 969.683
Expected volume = 1055.73
Relative error (expected) = 0.0815047
Relative error (exact) = 0.0530439
--- Testing volume of H-cube20
Number type: d
Computed volume 1.07218e+06
Expected volume = 1.02978e+06
Relative error (expected) = 0.0411754
Relative error (exact) = 0.022512
Computed volume 1.05618e+06
Expected volume = 1.05618e+06
Relative error (expected) = 2.67147e-06
Relative error (exact) = 0.00725443
Computed volume 1.04555e+06
Expected volume = 1.05883e+06
Relative error (expected) = 0.0125381
Relative error (exact) = 0.00288169
Computed volume 1.04555e+06
Expected volume = 1.05883e+06
Relative error (expected) = 0.0125381
Relative error (exact) = 0.00288169
--- Testing volume of H-cross10
Number type: d
Computed volume 0.000290503
Expected volume = 0.000283788
Relative error (expected) = 0.0236632
Relative error (exact) = 0.0294714
Computed volume 0.000280815
Expected volume = 0.000280815
Relative error (expected) = 2.44511e-07
Relative error (exact) = 0.00486191
Computed volume 0.000287524
Expected volume = 0.000296745
Relative error (expected) = 0.0310724
Relative error (exact) = 0.0189148
Computed volume 0.000287524
Expected volume = 0.000296745
Relative error (expected) = 0.0310724
Relative error (exact) = 0.0189148
--- Testing volume of H-birk3
Number type: d
Computed volume 0.120708
Expected volume = 0.130806
Relative error (expected) = 0.0772018
Relative error (exact) = 0.0343396
Computed volume 0.126776
Expected volume = 0.126776
Relative error (expected) = 1.0525e-07
Relative error (exact) = 0.0142079
Computed volume 0.119489
Expected volume = 0.122177
Relative error (expected) = 0.0220023
Relative error (exact) = 0.0440894
Computed volume 0.119489
Expected volume = 0.122177
Relative error (expected) = 0.0220023
Relative error (exact) = 0.0440894
--- Testing volume of H-birk4
Number type: d
Computed volume 0.000935426
Expected volume = 0.00112925
Relative error (expected) = 0.17164
Relative error (exact) = 0.0356614
Computed volume 0.000898527
Expected volume = 0.000898527
Relative error (expected) = 3.13174e-07
Relative error (exact) = 0.073701
Computed volume 0.000936353
Expected volume = 0.000945447
Relative error (expected) = 0.00961894
Relative error (exact) = 0.0347058
Computed volume 0.000936353
Expected volume = 0.000945447
Relative error (expected) = 0.00961894
Relative error (exact) = 0.0347058
--- Testing volume of H-birk5
Number type: d
Computed volume 1.75176e-07
Expected volume = 1.75176e-07
Relative error (expected) = 1.28417e-06
Relative error (exact) = 0.221441
Computed volume 2.07943e-07
Expected volume = 2.07943e-07
Relative error (expected) = 1.68918e-06
Relative error (exact) = 0.0758105
Computed volume 2.55711e-07
Expected volume = 2.80779e-07
Relative error (expected) = 0.0892798
Relative error (exact) = 0.136494
Computed volume 2.55711e-07
Expected volume = 2.80779e-07
Relative error (expected) = 0.0892798
Relative error (exact) = 0.136494
--- Testing volume of H-birk6
Number type: d
Computed volume 5.71076e-13
Expected volume = 5.71076e-13
Relative error (expected) = 6.16539e-07
Relative error (exact) = 0.396036
Computed volume 9.55284e-13
Expected volume = 9.48912e-13
Relative error (expected) = 0.00671524
Relative error (exact) = 0.0102991
Computed volume 8.47566e-13
Expected volume = 6.96476e-13
Relative error (expected) = 0.216936
Relative error (exact) = 0.103622
Computed volume 8.47566e-13
Expected volume = 6.96476e-13
Relative error (expected) = 0.216936
Relative error (exact) = 0.103622
--- Testing volume of H-prod_simplex5
Number type: d
Computed volume 7.03584e-05
Expected volume = 7.35223e-05
Relative error (expected) = 0.0430325
Relative error (exact) = 0.0131617
Computed volume 6.86576e-05
Expected volume = 6.86576e-05
Relative error (expected) = 4.34839e-07
Relative error (exact) = 0.011331
Computed volume 6.78605e-05
Expected volume = 7.43136e-05
Relative error (expected) = 0.0868366
Relative error (exact) = 0.0228094
Computed volume 6.78605e-05
Expected volume = 7.43136e-05
Relative error (expected) = 0.0868366
Relative error (exact) = 0.0228094
--- Testing volume of H-prod_simplex10
Number type: d
Computed volume 7.36178e-14
Expected volume = 7.38925e-14
Relative error (expected) = 0.00371704
Relative error (exact) = 0.0305864
Computed volume 8.01351e-14
Expected volume = 8.01351e-14
Relative error (expected) = 2.69799e-07
Relative error (exact) = 0.0552345
Computed volume 7.92832e-14
Expected volume = 8.27387e-14
Relative error (expected) = 0.0417642
Relative error (exact) = 0.044016
Computed volume 7.92832e-14
Expected volume = 8.27387e-14
Relative error (expected) = 0.0417642
Relative error (exact) = 0.044016
--- Testing volume of H-prod_simplex15
Number type: d
Computed volume 5.57339e-25
Expected volume = 5.61238e-25
Relative error (expected) = 0.00694784
Relative error (exact) = 0.0469441
Computed volume 5.87558e-25
Expected volume = 5.87558e-25
Relative error (expected) = 7.62152e-07
Relative error (exact) = 0.00473061
Computed volume 6.14854e-25
Expected volume = 5.48179e-25
Relative error (expected) = 0.12163
Relative error (exact) = 0.0514076
Computed volume 6.14854e-25
Expected volume = 5.48179e-25
Relative error (expected) = 0.12163
Relative error (exact) = 0.0514076
--- Testing volume of H-simplex10
Number type: d
Computed volume 2.74631e-07
Expected volume = 2.98074e-07
Relative error (expected) = 0.0786493
Relative error (exact) = 0.00342012
Computed volume 2.52756e-07
Expected volume = 2.52756e-07
Relative error (expected) = 1.43151e-06
Relative error (exact) = 0.0827977
Computed volume 2.81579e-07
Expected volume = 2.89366e-07
Relative error (expected) = 0.0269097
Relative error (exact) = 0.0217948
Computed volume 2.81579e-07
Expected volume = 2.89366e-07
Relative error (expected) = 0.0269097
Relative error (exact) = 0.0217948
--- Testing volume of H-simplex20
Number type: d
Computed volume 4.13723e-19
Expected volume = 4.64611e-19
Relative error (expected) = 0.109527
Relative error (exact) = 0.00654858
Computed volume 4.4317e-19
Expected volume = 4.4317e-19
Relative error (expected) = 4.54657e-07
Relative error (exact) = 0.0781897
Computed volume 4.12885e-19
Expected volume = 4.16737e-19
Relative error (expected) = 0.00924392
Relative error (exact) = 0.00450806
Computed volume 4.12885e-19
Expected volume = 4.16737e-19
Relative error (expected) = 0.00924392
Relative error (exact) = 0.00450806
--- Testing volume of H-simplex30
Number type: d
Computed volume 3.91407e-33
Expected volume = 3.65853e-33
Relative error (expected) = 0.0698486
Relative error (exact) = 0.0382191
Computed volume 3.86474e-33
Expected volume = 3.86474e-33
Relative error (expected) = 8.17562e-07
Relative error (exact) = 0.0251342
Computed volume 3.37844e-33
Expected volume = 4.04136e-33
Relative error (expected) = 0.164034
Relative error (exact) = 0.103859
Computed volume 3.37844e-33
Expected volume = 4.04136e-33
Relative error (expected) = 0.164034
Relative error (exact) = 0.103859
--- Testing volume of H-skinny_cube10
===============================================================================
[doctest] test cases:      6 |      6 passed |      0 failed |      0 skipped
[doctest] assertions:     52 |     52 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

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
    CHECK((std::abs((volume - exact)/exact) < 0.35 || 
           std::abs((volume - expected)/expected) < 0.00001));
}

template <class Polytope>
void test_volume(Polytope &HP,
                 double const& expectedBall,
                 double const& expectedCDHR,
                 double const& expectedRDHR,
                 double const& expectedBilliard,
                 double const& exact,
                 bool birk = false)
{
    typedef typename Polytope::PointType Point;
    typedef typename Point::FT NT;

    // Setup the parameters
    int walk_len = 10 + HP.dimension()/10;
    NT e=0.1, volume;

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 3> RNGType;

    //TODO: low accuracy in high dimensions
    if (!birk) {
        volume = volume_cooling_balls<BallWalk, RNGType>(HP, e, walk_len).second;
        test_values(volume, expectedBall, exact);
    }

    volume = volume_cooling_balls<CDHRWalk, RNGType>(HP, e, walk_len).second;
    test_values(volume, expectedCDHR, exact);

    volume = volume_cooling_balls<RDHRWalk, RNGType>(HP, e, walk_len).second;
    test_values(volume, expectedRDHR, exact);

    volume = volume_cooling_balls<BilliardWalk, RNGType>(HP, e, walk_len).second;
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
    test_volume(P, 1118.63, 1163.36, 1119.15, 1100.73, 1024);

    std::cout << "--- Testing volume of H-cube20" << std::endl;
    P = generate_cube<Hpolytope>(20, false);
    test_volume(P, 965744, 1051230, 1006470, 1007020, 1048576);
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
                0.000291034,
                0.000281135,
                0.000294805,
                0.000286491,
                0.0002821869);
}

template <typename NT>
void call_test_birk()
{
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef HPolytope<Point> Hpolytope;
    Hpolytope P;

    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 123> RNGType;

    std::cout << "--- Testing volume of H-birk3" << std::endl;
    P = generate_birkhoff<Hpolytope>(3);
    test_volume(P, 0.114343, 0.125548, 0.113241, 0.112446, 0.125, true);

    std::cout << "--- Testing volume of H-birk4" << std::endl;
    P = generate_birkhoff<Hpolytope>(4);
    test_volume(P, 0.00112956, 0.00109593, 0.00108152, 0.000845192,
                0.000970018, true);

    std::cout << "--- Testing volume of H-birk5" << std::endl;
    P = generate_birkhoff<Hpolytope>(5);
    test_volume(P,
                1.97968e-07,
                1.73729e-07,
                1.39042e-07,
                3.24308e-07,
                0.000000225, 
                true);

    std::cout << "--- Testing volume of H-birk6" << std::endl;
    P = generate_birkhoff<Hpolytope>(6);
    test_volume(P,
                7.84351e-13,
                6.10783e-13,
                2.7408e-13,
                6.62349e-13,
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
                6.40072 * std::pow(10,-5),
                6.69062 * std::pow(10,-5),
                6.20744e-05,
                6.31986 * std::pow(10,-5),
                std::pow(1.0 / factorial(5.0), 2));

    std::cout << "--- Testing volume of H-prod_simplex10" << std::endl;
    P = generate_prod_simplex<Hpolytope>(10);
    test_volume(P,
                6.83631 * std::pow(10,-14),
                8.19581 * std::pow(10,-14),
                9.35005e-14,
                6.57309e-14,
                std::pow(1.0 / factorial(10.0), 2));

    std::cout << "--- Testing volume of H-prod_simplex15" << std::endl;
    P = generate_prod_simplex<Hpolytope>(15);
    test_volume(P,
                3.85153e-25,
                9.33162 * std::pow(10,-25),
                3.95891e-25,
                5.72542e-25,
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
                3.90133e-07,
                2.90617 * std::pow(10,-7),
                2.93392 * std::pow(10,-7),
                3.00286e-07,
                1.0 / factorial(10.0));

    std::cout << "--- Testing volume of H-simplex20" << std::endl;
    P = generate_simplex<Hpolytope>(20, false);
    test_volume(P,
                6.52535e-19,
                4.14182 * std::pow(10,-19),
                4.5877e-19,
                4.54245e-19,
                1.0 / factorial(20.0));

    std::cout << "--- Testing volume of H-simplex30" << std::endl;
    P = generate_simplex<Hpolytope>(30, false);
    test_volume(P,
                2.5776 * std::pow(10,-33),
                3.5157 * std::pow(10,-33),
                2.74483e-33,
                3.08769e-33,
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
Computed volume 1092.09
Expected volume = 1118.63
Relative error (expected) = 0.0237284
Relative error (exact) = 0.0664909
Computed volume 1060.98
Expected volume = 1163.36
Relative error (expected) = 0.0880028
Relative error (exact) = 0.0361143
Computed volume 1110.3
Expected volume = 1119.15
Relative error (expected) = 0.0079035
Relative error (exact) = 0.084282
Computed volume 1102.25
Expected volume = 1100.73
Relative error (expected) = 0.00137887
Relative error (exact) = 0.0764138
--- Testing volume of H-cube20
Number type: d
Computed volume 956389
Expected volume = 965744
Relative error (expected) = 0.00968665
Relative error (exact) = 0.0879162
Computed volume 982563
Expected volume = 1.05123e+06
Relative error (expected) = 0.0653209
Relative error (exact) = 0.0629552
Computed volume 922887
Expected volume = 1.00647e+06
Relative error (expected) = 0.0830459
Relative error (exact) = 0.119867
Computed volume 995298
Expected volume = 1.00702e+06
Relative error (expected) = 0.0116402
Relative error (exact) = 0.0508098
--- Testing volume of H-cross10
Number type: d
Computed volume 0.000306295
Expected volume = 0.000291034
Relative error (expected) = 0.0524355
Relative error (exact) = 0.0854313
Computed volume 0.000289388
Expected volume = 0.000281135
Relative error (expected) = 0.0293547
Relative error (exact) = 0.0255176
Computed volume 0.000307491
Expected volume = 0.000294805
Relative error (expected) = 0.0430323
Relative error (exact) = 0.0896719
Computed volume 0.000291618
Expected volume = 0.000286491
Relative error (expected) = 0.0178949
Relative error (exact) = 0.0334205
--- Testing volume of H-birk3
Number type: d
Computed volume 0.125674
Expected volume = 0.125548
Relative error (expected) = 0.000999805
Relative error (exact) = 0.00538819
Computed volume 0.110981
Expected volume = 0.113241
Relative error (expected) = 0.0199595
Relative error (exact) = 0.112154
Computed volume 0.108596
Expected volume = 0.112446
Relative error (expected) = 0.0342353
Relative error (exact) = 0.131229
--- Testing volume of H-birk4
Number type: d
Computed volume 0.0008604
Expected volume = 0.00109593
Relative error (expected) = 0.214914
Relative error (exact) = 0.113007
Computed volume 0.000884399
Expected volume = 0.00108152
Relative error (expected) = 0.182263
Relative error (exact) = 0.0882655
Computed volume 0.000912338
Expected volume = 0.000845192
Relative error (expected) = 0.079445
Relative error (exact) = 0.0594625
--- Testing volume of H-birk5
Number type: d
Computed volume 2.16486e-07
Expected volume = 1.73729e-07
Relative error (expected) = 0.246116
Relative error (exact) = 0.0378382
Computed volume 2.26005e-07
Expected volume = 1.39042e-07
Relative error (expected) = 0.625446
Relative error (exact) = 0.00446788
Computed volume 2.81101e-07
Expected volume = 3.24308e-07
Relative error (expected) = 0.133229
Relative error (exact) = 0.249337
--- Testing volume of H-birk6
Number type: d
Computed volume 7.27437e-13
Expected volume = 6.10783e-13
Relative error (expected) = 0.190991
Relative error (exact) = 0.23067
Computed volume 2.7408e-13
Expected volume = 2.7408e-13
Relative error (expected) = 1.48642e-06
Relative error (exact) = 0.710136
Computed volume 1.01884e-12
Expected volume = 6.62349e-13
Relative error (expected) = 0.538218
Relative error (exact) = 0.0775121
--- Testing volume of H-prod_simplex5
Number type: d
Computed volume 6.97656e-05
Expected volume = 6.40072e-05
Relative error (expected) = 0.0899646
Relative error (exact) = 0.00462437
Computed volume 6.62786e-05
Expected volume = 6.69062e-05
Relative error (expected) = 0.00938068
Relative error (exact) = 0.0455885
Computed volume 7.34139e-05
Expected volume = 6.20744e-05
Relative error (expected) = 0.182675
Relative error (exact) = 0.0571595
Computed volume 6.5947e-05
Expected volume = 6.31986e-05
Relative error (expected) = 0.0434877
Relative error (exact) = 0.0503638
--- Testing volume of H-prod_simplex10
Number type: d
Computed volume 5.32259e-14
Expected volume = 6.83631e-14
Relative error (expected) = 0.221424
Relative error (exact) = 0.299112
Computed volume 8.5011e-14
Expected volume = 8.19581e-14
Relative error (expected) = 0.0372501
Relative error (exact) = 0.119442
Computed volume 7.52589e-14
Expected volume = 9.35005e-14
Relative error (expected) = 0.195096
Relative error (exact) = 0.00897611
Computed volume 7.74301e-14
Expected volume = 6.57309e-14
Relative error (expected) = 0.177986
Relative error (exact) = 0.019614
--- Testing volume of H-prod_simplex15
Number type: d
Computed volume 3.85153e-25
Expected volume = 3.85153e-25
Relative error (expected) = 8.6892e-07
Relative error (exact) = 0.341384
Computed volume 5.19882e-25
Expected volume = 9.33162e-25
Relative error (expected) = 0.442881
Relative error (exact) = 0.110995
Computed volume 4.90706e-25
Expected volume = 3.95891e-25
Relative error (expected) = 0.239498
Relative error (exact) = 0.160886
Computed volume 6.0693e-25
Expected volume = 5.72542e-25
Relative error (expected) = 0.0600624
Relative error (exact) = 0.0378581
--- Testing volume of H-simplex10
Number type: d
Computed volume 3.90133e-07
Expected volume = 3.90133e-07
Relative error (expected) = 6.03668e-07
Relative error (exact) = 0.415714
Computed volume 3.0417e-07
Expected volume = 2.90617e-07
Relative error (expected) = 0.0466351
Relative error (exact) = 0.103772
Computed volume 3.55006e-07
Expected volume = 2.93392e-07
Relative error (expected) = 0.210006
Relative error (exact) = 0.288246
Computed volume 3.05323e-07
Expected volume = 3.00286e-07
Relative error (expected) = 0.0167743
Relative error (exact) = 0.107956
--- Testing volume of H-simplex20
Number type: d
Computed volume 6.52535e-19
Expected volume = 6.52535e-19
Relative error (expected) = 3.67013e-07
Relative error (exact) = 0.587553
Computed volume 4.13341e-19
Expected volume = 4.14182e-19
Relative error (expected) = 0.00202994
Relative error (exact) = 0.00561872
Computed volume 4.43975e-19
Expected volume = 4.5877e-19
Relative error (expected) = 0.0322497
Relative error (exact) = 0.0801471
Computed volume 3.82222e-19
Expected volume = 4.54245e-19
Relative error (expected) = 0.158555
Relative error (exact) = 0.0700914
--- Testing volume of H-simplex30
Number type: d
Computed volume 2.70368e-33
Expected volume = 2.5776e-33
Relative error (expected) = 0.0489148
Relative error (exact) = 0.28284
Computed volume 3.72868e-33
Expected volume = 3.5157e-33
Relative error (expected) = 0.060579
Relative error (exact) = 0.0109576
Computed volume 4.34597e-33
Expected volume = 2.74483e-33
Relative error (expected) = 0.583328
Relative error (exact) = 0.15278
Computed volume 3.26746e-33
Expected volume = 3.08769e-33
Relative error (expected) = 0.0582215
Relative error (exact) = 0.133297
--- Testing volume of H-skinny_cube10
===============================================================================
[doctest] test cases:      6 |      6 passed |      0 failed |      0 skipped
[doctest] assertions:     48 |     48 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

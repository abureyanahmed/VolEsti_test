#include "doctest.h"
#include "exact_vols.h"
#include <fstream>
#include <iostream>
#include "misc.h"
#include "random.hpp"
#include "random/uniform_int.hpp"
#include "random/normal_distribution.hpp"
#include "random/uniform_real_distribution.hpp"
#include "random_walks/random_walks.hpp"
#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "volume/volume_cooling_hpoly.hpp"
#include "z_polytopes_generators.h"


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
void test_volume_hpoly(Polytope &P,
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
    typedef HPolytope<Point> Hpolytope;

    // Setup the parameters
    int walk_len = 1, dim = P.dimension();
    NT e = 0.1, volume;
    MT G = P.get_mat();
    VT b = P.get_vec();

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 5> RNGType;

    //TODO: low accuracy in high dimensions
    //NT volume = volume_cooling_balls<BallWalk, RNGType>(HP, e, walk_len);
    //test_values(volume, expectedBall, exact);
    Polytope P1(dim, G, b);
    volume = volume_cooling_hpoly<CDHRWalk, RNGType, Hpolytope>(P1, e, walk_len);
    test_values(volume, expectedCDHR, exact);

    Polytope P2(dim, G, b);
    volume = volume_cooling_hpoly<RDHRWalk, RNGType, Hpolytope>(P2, e, walk_len);
    test_values(volume, expectedRDHR, exact);

    Polytope P3(dim, G, b);
    volume = volume_cooling_hpoly<BilliardWalk, RNGType, Hpolytope>(P3, e, walk_len);
    test_values(volume, expectedBilliard, exact);
}

template <class Polytope>
void test_volume_balls(Polytope &P,
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
    typedef HPolytope<Point> Hpolytope;

    // Setup the parameters
    int walk_len = 1, dim = P.dimension();
    NT e = 0.1, volume;
    MT G = P.get_mat();
    VT b = P.get_vec();

    // Estimate the volume
    std::cout << "Number type: " << typeid(NT).name() << std::endl;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT, 5> RNGType;

    Polytope P1(dim, G, b);
    volume = volume_cooling_balls<CDHRWalk, RNGType>(P1, e, walk_len).second;
    test_values(volume, expectedCDHR, exact);

    Polytope P2(dim, G, b);
    volume = volume_cooling_balls<RDHRWalk, RNGType>(P2, e, walk_len).second;
    test_values(volume, expectedRDHR, exact);

    Polytope P3(dim, G, b);
    volume = volume_cooling_balls<BilliardWalk, RNGType>(P3, e, walk_len).second;
    test_values(volume, expectedBilliard, exact);
}


template <typename NT>
void call_test_uniform_generator(){
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef boost::mt19937            RNGType;
    typedef Zonotope<Point> zonotope;
    zonotope P;

    P = gen_zonotope_uniform<zonotope, RNGType>(5, 10, 127);
    NT exact_vol = exact_zonotope_vol<NT>(P);
    test_volume_hpoly(P, exact_vol, exact_vol, exact_vol, exact_vol, exact_vol);
    test_volume_balls(P, exact_vol, exact_vol, exact_vol, exact_vol, exact_vol);

    P = gen_zonotope_uniform<zonotope, RNGType>(10, 15, 211);
    exact_vol = exact_zonotope_vol<NT>(P);
    test_volume_hpoly(P,
                      0,
                      6.95342e+20,
                      7.27889 * std::pow(10,20),
                      7.18605 * std::pow(10,20),
                      exact_vol);
    test_volume_balls(P,
                      0,
                      3.42945 * std::pow(10,20),
                      4.68065 * std::pow(10,20),
                      6.45698 * std::pow(10,20),
                      exact_vol);
}


TEST_CASE("uniform_zonotopes") {
    call_test_uniform_generator<double>();
    //call_test_cube_float<float>();
}


/*

[doctest] doctest version is "1.2.9"
[doctest] run with "--help" for options
Number type: d
Computed volume 1.189e+11
Expected volume = 1.13244e+11
Relative error (expected) = 0.0499462
Relative error (exact) = 0.0499462
Computed volume 1.15474e+11
Expected volume = 1.13244e+11
Relative error (expected) = 0.0196897
Relative error (exact) = 0.0196897
Computed volume 1.06685e+11
Expected volume = 1.13244e+11
Relative error (expected) = 0.0579179
Relative error (exact) = 0.0579179
Number type: d
Computed volume 1.12147e+11
Expected volume = 1.13244e+11
Relative error (expected) = 0.00969006
Relative error (exact) = 0.00969006
Computed volume 1.07192e+11
Expected volume = 1.13244e+11
Relative error (expected) = 0.0534391
Relative error (exact) = 0.0534391
Computed volume 1.07487e+11
Expected volume = 1.13244e+11
Relative error (expected) = 0.05084
Relative error (exact) = 0.05084
Number type: d
Computed volume 6.95342e+20
Expected volume = 6.95342e+20
Relative error (expected) = 2.36198e-07
Relative error (exact) = 0.0778302
Computed volume 6.98637e+20
Expected volume = 7.27889e+20
Relative error (expected) = 0.040188
Relative error (exact) = 0.0829369
Computed volume 7.18605e+20
Expected volume = 7.18605e+20
Relative error (expected) = 4.05789e-07
Relative error (exact) = 0.11389
Number type: d
Computed volume 3.42945e+20
Expected volume = 3.42945e+20
Relative error (expected) = 4.15062e-07
Relative error (exact) = 0.46841
Computed volume 4.68065e+20
Expected volume = 4.68065e+20
Relative error (expected) = 4.65377e-07
Relative error (exact) = 0.274466
Computed volume 6.45698e+20
Expected volume = 6.45698e+20
Relative error (expected) = 5.30117e-07
Relative error (exact) = 0.000877659
===============================================================================
[doctest] test cases:      1 |      1 passed |      0 failed |      0 skipped
[doctest] assertions:     12 |     12 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

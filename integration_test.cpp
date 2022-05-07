#include "cartesian_geom/cartesian_kernel.h"
#include "convex_bodies/hpolytope.h"
#include "doctest.h"
#include "Eigen/Eigen"
#include <fstream>
#include "generators/known_polytope_generators.h"
#include <iostream>
#include "misc.h"
#include "ode_solvers/oracle_functors.hpp"
#include "random_walks/random_walks.hpp"
#include "simple_MC_integration.hpp"
#include <vector>

template <typename NT>
NT exp_normsq(Point X) {
	return exp(-X.squared_length()) ;
}

template <typename NT>
NT simple_polynomial_1D(Point X) {
	return (X[0] - 1) * (X[0] - 2) * (X[0] - 3);
}

template <typename NT>
NT logx_natural_1D(Point X) {
	return log(X[0]);
}

template <typename NT>
NT rooted_squaresum(Point X) {
	return sqrt(X.squared_length());
}

template <typename NT>
NT one_sqsum(Point X) {
	return 1 - X.squared_length();
}

template <typename NT>
void test_values (NT computed, NT expected, NT exact) {
	std::cout << "Computed integration value = " << computed << std::endl;
	std::cout << "Expected integration value = " << expected << std::endl;
	std::cout << "Exact integration value = " << exact << std::endl;
	std::cout << "Relative error (expected) = " << std::abs((computed - expected)/expected) << std::endl;
	std::cout << "Relative error (exact) = " << std::abs((computed - exact)/exact) << std::endl ;
    CHECK(((std::abs((computed - expected)/expected) < 0.00001) || (std::abs((computed - exact)/exact) < 0.2)));
}

template <typename NT>
void  call_test_simple_mc_integration_over_rectangles() {

	typedef Cartesian<NT> Kernel;
	typedef typename Kernel::Point Point;
	typedef HPolytope<Point> HPOLYTOPE;
	typedef VPolytope<Point> VPOLYTOPE;
	typedef boost::mt19937 RNGType;
	typedef BoostRandomNumberGenerator<RNGType, NT> RandomNumberGenerator;
	
	NT integration_value;
	std::cout << "\nTESTS FOR SIMPLE MC INTEGRATION OVER RECTANGLES USING UNIFORM RANDOM WALKS\n";

	Limit LL{-1};
	Limit UL{6};
	integration_value = simple_mc_integrate <AcceleratedBilliardWalk> (simple_polynomial_1D<NT>, 1, 100000, CB, LL, UL);
	test_values(integration_value, 39.7, 40.25);

	Limit LL1{0.5};
	Limit UL1{10};
	integration_value = simple_mc_integrate <BilliardWalk> (logx_natural_1D<NT>, 1, 1000, CB, LL1, UL1);
	test_values(integration_value, 13.65, 13.872);

	Limit LL2{-1, -1};
	Limit UL2{1, 1};
	integration_value = simple_mc_integrate <AcceleratedBilliardWalk> (rooted_squaresum<NT>, 2, 1000, SOB, LL2, UL2);
	test_values(integration_value, 2.99, 3.0607);

	integration_value = simple_mc_integrate <BilliardWalk> (exp_normsq<NT>, 5, 1000, SOB);
	test_values(integration_value, 7.49, 7.46);

	integration_value = simple_mc_integrate <AcceleratedBilliardWalk> (exp_normsq<NT>, 8, 1000, SOB);
	test_values(integration_value, 24.8, 24.76);

	integration_value = simple_mc_integrate <BilliardWalk> (exp_normsq<NT>, 10, 10000, SOB);
	test_values(integration_value, 54.8, 55.25);

}

template <typename NT>
void call_test_simple_mc_integration_over_cubes() {

	typedef Cartesian<NT> Kernel;
	typedef typename Kernel::Point Point;
	typedef HPolytope<Point> HPOLYTOPE;
	typedef VPolytope<Point> VPOLYTOPE;
	typedef boost::mt19937 RNGType;
	typedef BoostRandomNumberGenerator<RNGType, NT> RandomNumberGenerator;

	std::cout << "\nTESTS FOR SIMPLE MC INTEGRATION OVER CUBES USING UNIFORM RANDOM WALKS\n";

	NT integration_value;
	HPOLYTOPE HP;

	HP = generate_cube <HPOLYTOPE> (2, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (exp_normsq<NT>, HP, 1000, SOB, 10, 0.01);
	test_values(integration_value, 2.20, 2.230);

	// For 2D Polytope shifted to (1,1) from origin
	std::vector<NT> Origin{1, 1};
	Point newOrigin(2, Origin);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (exp_normsq<NT>, HP, 1000, SOB, 1, 0.01, newOrigin);
	test_values(integration_value, 0.78, 0.777);

	HP = generate_cube <HPOLYTOPE> (10, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (exp_normsq<NT>, HP, 10000, SOB);
	test_values(integration_value, 54.7, 55.25);

	HP = generate_cube <HPOLYTOPE> (15, false);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (exp_normsq<NT>, HP, 10000, SOB);
	test_values(integration_value, 405.9, 410.690);

	HP = generate_cube <HPOLYTOPE> (20, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (exp_normsq<NT>, HP, 10000, SOB);
	test_values(integration_value, 3050.0, 3052.71);

	// Reading a H-Polytope from *.ine file for 20 Dimensions
	// std::string fileName("cube10.ine");
	// std::ifstream inp;
	// std::vector<std::vector<NT>> Pin;
	// inp.open(fileName, std::ifstream::in);
	// read_pointset(inp,Pin);
	// HPOLYTOPE HP(Pin);
	// integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (exp_normsq<NT>, 2, HP, 1000, SOB);
	// std::cout << "Integration value: " << integration_value << std::endl;
	// test_values(integration_value, expected, exact);
	// inp.close();
}

template <typename NT>
void call_test_simple_mc_integration_over_simplices() {

	typedef Cartesian<NT> Kernel;
	typedef typename Kernel::Point Point;
	typedef HPolytope<Point> HPOLYTOPE;
	typedef VPolytope<Point> VPOLYTOPE;
	typedef boost::mt19937 RNGType;
	typedef BoostRandomNumberGenerator<RNGType, NT> RandomNumberGenerator;

	std::cout << "\nTESTS FOR SIMPLE MC INTEGRATION OVER SIMPLICES USING UNIFORM RANDOM WALKS\n";

	NT integration_value;
	HPOLYTOPE HP;

	HP = generate_simplex <HPOLYTOPE> (1, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, CB, 10, 0.01);
	test_values(integration_value, 0.67, 0.666);
	
	HP = generate_simplex <HPOLYTOPE> (2, false);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB, 10, 0.01);
	test_values(integration_value, 0.34, 0.333);

	HP = generate_simplex <HPOLYTOPE> (3, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB);
	test_values(integration_value, 0.116, 0.1166);

	HP = generate_simplex <HPOLYTOPE> (5, false);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB);
	test_values(integration_value, 0.00656, 0.0063492);

	HP = generate_simplex <HPOLYTOPE> (7, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 10000, SOB);
	test_values(integration_value, 0.000159, 0.000159832);

}

template <typename NT>
void call_test_simple_mc_integration_over_product_simplices() {

	typedef Cartesian<NT> Kernel;
	typedef typename Kernel::Point Point;
	typedef HPolytope<Point> HPOLYTOPE;
	typedef VPolytope<Point> VPOLYTOPE;
	typedef boost::mt19937 RNGType;
	typedef BoostRandomNumberGenerator<RNGType, NT> RandomNumberGenerator;

	std::cout << "\nTESTS FOR SIMPLE MC INTEGRATION OVER PRODUCT SIMPLICES USING UNIFORM RANDOM WALKS\n";

	NT integration_value;
	HPOLYTOPE HP;

	HP = generate_prod_simplex <HPOLYTOPE> (1, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, CB, 10, 0.01);
	test_values(integration_value, 0.334, 0.333);
	
	HP = generate_prod_simplex <HPOLYTOPE> (2, false);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB);
	test_values(integration_value, 0.0834, 0.0833);

	HP = generate_prod_simplex <HPOLYTOPE> (3, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB);
	test_values(integration_value, 0.0110, 0.01111);

	HP = generate_prod_simplex <HPOLYTOPE> (5, false);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 10000, SOB);
	test_values(integration_value, 0.36e-4, 0.36375e-4);

	HP = generate_prod_simplex <HPOLYTOPE> (7, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>,  HP, 10000, SOB);
	test_values(integration_value, 0.235e-7, 0.24079e-7);

}

template <typename NT>
void call_test_simple_mc_integration_over_cross_polytopes() {

	typedef Cartesian<NT> Kernel;
	typedef typename Kernel::Point Point;
	typedef HPolytope<Point> HPOLYTOPE;
	typedef VPolytope<Point> VPOLYTOPE;
	typedef boost::mt19937 RNGType;
	typedef BoostRandomNumberGenerator<RNGType, NT> RandomNumberGenerator;

	std::cout << "\nTESTS FOR SIMPLE MC INTEGRATION OVER CROSS POLYTOPES USING UNIFORM RANDOM WALKS\n";

	NT integration_value;
	HPOLYTOPE HP;

	HP = generate_cross <HPOLYTOPE> (1, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, CB, 10, 0.01);
	test_values(integration_value, 1.334, 1.333333);
	
	HP = generate_cross <HPOLYTOPE> (2, false);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB, 10, 0.01);
	test_values(integration_value, 1.334, 1.33333);

	HP = generate_cross <HPOLYTOPE> (3, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB);
	test_values(integration_value, 0.935000, 0.933333);

	HP = generate_cross <HPOLYTOPE> (5, false);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB);
	test_values(integration_value, 0.200000, 0.203174);

	HP = generate_cross <HPOLYTOPE> (7, false);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 10000, SOB);
	test_values(integration_value, 0.020000, 0.020458);

}

template <typename NT>
void call_test_simple_mc_integration_over_birkhoff_polytopes() {

	typedef Cartesian<NT> Kernel;
	typedef typename Kernel::Point Point;
	typedef HPolytope<Point> HPOLYTOPE;
	typedef VPolytope<Point> VPOLYTOPE;
	typedef boost::mt19937 RNGType;
	typedef BoostRandomNumberGenerator<RNGType, NT> RandomNumberGenerator;

	std::cout << "\nTESTS FOR SIMPLE MC INTEGRATION OVER BIRKHOFF POLYTOPES USING UNIFORM RANDOM WALKS\n";

	NT integration_value;
	HPOLYTOPE HP;

	HP = generate_birkhoff <HPOLYTOPE> (2);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, CB, 10, 0.01);
	test_values(integration_value, 0.67, 0.6666);

	HP = generate_birkhoff <HPOLYTOPE> (3);
	integration_value = simple_mc_polytope_integrate <AcceleratedBilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 1000, SOB);
	test_values(integration_value, 0.0470, 0.04722);
	
	HP = generate_birkhoff <HPOLYTOPE> (4);
	integration_value = simple_mc_polytope_integrate <BilliardWalk, HPOLYTOPE> (one_sqsum<NT>, HP, 10000, SOB);
	test_values(integration_value, 0.000150, 0.000164);

}

TEST_CASE("rectangle") {
    call_test_simple_mc_integration_over_rectangles<double>();
}

TEST_CASE("cube") {
    call_test_simple_mc_integration_over_cubes<double>();
}

TEST_CASE("simplex") {
    call_test_simple_mc_integration_over_simplices<double>();
}

TEST_CASE("prod_simplex") {
	call_test_simple_mc_integration_over_product_simplices<double>();
}

TEST_CASE("cross") {
	call_test_simple_mc_integration_over_cross_polytopes<double>();
}

TEST_CASE("birkhoff") {
	call_test_simple_mc_integration_over_birkhoff_polytopes<double>();
}

/*

[doctest] doctest version is "1.2.9"
[doctest] run with "--help" for options

TESTS FOR SIMPLE MC INTEGRATION OVER RECTANGLES USING UNIFORM RANDOM WALKS
Computed integration value = 36.1111
Expected integration value = 39.7
Exact integration value = 40.25
Relative error (expected) = 0.0903999
Relative error (exact) = 0.102829
Computed integration value = 15
Expected integration value = 13.65
Exact integration value = 13.872
Relative error (expected) = 0.0989027
Relative error (exact) = 0.0813164
Computed integration value = 3.08113
Expected integration value = 2.99
Exact integration value = 3.0607
Relative error (expected) = 0.0304768
Relative error (exact) = 0.00667352
Computed integration value = 7.57213
Expected integration value = 7.49
Exact integration value = 7.46
Relative error (expected) = 0.0109648
Relative error (exact) = 0.0150303
Computed integration value = 25.3586
Expected integration value = 24.8
Exact integration value = 24.76
Relative error (expected) = 0.0225223
Relative error (exact) = 0.0241742
Computed integration value = 55.4401
Expected integration value = 54.8
Exact integration value = 55.25
Relative error (expected) = 0.0116804
Relative error (exact) = 0.00344045

TESTS FOR SIMPLE MC INTEGRATION OVER CUBES USING UNIFORM RANDOM WALKS
Computed integration value = 2.24134
Expected integration value = 2.2
Exact integration value = 2.23
Relative error (expected) = 0.0187895
Relative error (exact) = 0.00508381
Computed integration value = 0.812326
Expected integration value = 0.78
Exact integration value = 0.777
Relative error (expected) = 0.0414437
Relative error (exact) = 0.0454647
Computed integration value = 56.2765
Expected integration value = 54.7
Exact integration value = 55.25
Relative error (expected) = 0.0288209
Relative error (exact) = 0.0185792
Computed integration value = 411.71
Expected integration value = 405.9
Exact integration value = 410.69
Relative error (expected) = 0.0143132
Relative error (exact) = 0.00248295
Computed integration value = 3027.2
Expected integration value = 3050
Exact integration value = 3052.71
Relative error (expected) = 0.00747466
Relative error (exact) = 0.00835576

TESTS FOR SIMPLE MC INTEGRATION OVER SIMPLICES USING UNIFORM RANDOM WALKS
Computed integration value = 0.668483
Expected integration value = 0.67
Exact integration value = 0.666
Relative error (expected) = 0.00226345
Relative error (exact) = 0.00372896
Computed integration value = 0.3281
Expected integration value = 0.34
Exact integration value = 0.333
Relative error (expected) = 0.0349995
Relative error (exact) = 0.0147142
Computed integration value = 0.116553
Expected integration value = 0.116
Exact integration value = 0.1166
Relative error (expected) = 0.00477087
Relative error (exact) = 0.00039948
Computed integration value = 0.00625463
Expected integration value = 0.00656
Exact integration value = 0.0063492
Relative error (expected) = 0.0465505
Relative error (exact) = 0.014895
Computed integration value = 0.000158702
Expected integration value = 0.000159
Exact integration value = 0.000159832
Relative error (expected) = 0.00187122
Relative error (exact) = 0.00706695

TESTS FOR SIMPLE MC INTEGRATION OVER PRODUCT SIMPLICES USING UNIFORM RANDOM WALKS
Computed integration value = 0.329511
Expected integration value = 0.334
Exact integration value = 0.333
Relative error (expected) = 0.0134405
Relative error (exact) = 0.0104779
Computed integration value = 0.0812348
Expected integration value = 0.0834
Exact integration value = 0.0833
Relative error (expected) = 0.0259619
Relative error (exact) = 0.0247926
Computed integration value = 0.0115655
Expected integration value = 0.011
Exact integration value = 0.01111
Relative error (expected) = 0.0514123
Relative error (exact) = 0.0410023
Computed integration value = 3.61612e-05
Expected integration value = 3.6e-05
Exact integration value = 3.6375e-05
Relative error (expected) = 0.00447745
Relative error (exact) = 0.00587798
Computed integration value = 2.41134e-08
Expected integration value = 2.35e-08
Exact integration value = 2.4079e-08
Relative error (expected) = 0.026103
Relative error (exact) = 0.00142948

TESTS FOR SIMPLE MC INTEGRATION OVER CROSS POLYTOPES USING UNIFORM RANDOM WALKS
Computed integration value = 1.3821
Expected integration value = 1.334
Exact integration value = 1.33333
Relative error (expected) = 0.0360588
Relative error (exact) = 0.0365771
Computed integration value = 1.3344
Expected integration value = 1.334
Exact integration value = 1.33333
Relative error (expected) = 0.000297224
Relative error (exact) = 0.000799874
Computed integration value = 0.923907
Expected integration value = 0.935
Exact integration value = 0.933333
Relative error (expected) = 0.0118641
Relative error (exact) = 0.0100992
Computed integration value = 0.202019
Expected integration value = 0.2
Exact integration value = 0.203174
Relative error (expected) = 0.0100963
Relative error (exact) = 0.00568355
Computed integration value = 0.0205219
Expected integration value = 0.02
Exact integration value = 0.020458
Relative error (expected) = 0.0260932
Relative error (exact) = 0.00312169

TESTS FOR SIMPLE MC INTEGRATION OVER BIRKHOFF POLYTOPES USING UNIFORM RANDOM WALKS
Computed integration value = 0.688611
Expected integration value = 0.67
Exact integration value = 0.6666
Relative error (expected) = 0.0277773
Relative error (exact) = 0.0330194
Computed integration value = 0.0477066
Expected integration value = 0.047
Exact integration value = 0.04722
Relative error (expected) = 0.015035
Relative error (exact) = 0.0103059
Computed integration value = 0.000166561
Expected integration value = 0.00015
Exact integration value = 0.000164
Relative error (expected) = 0.110407
Relative error (exact) = 0.0156157
===============================================================================
[doctest] test cases:      6 |      6 passed |      0 failed |      0 skipped
[doctest] assertions:     29 |     29 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

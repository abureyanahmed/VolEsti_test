#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "doctest.h"
#include <fstream>
#include <iostream>
#include "misc.h"
#include "orderpolytope.h"
#include "poset.h"
#include "random.hpp"


template <typename NT>
void call_test_reflection() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Pnt;
    typedef typename OrderPolytope<Pnt>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Create Poset
    // The poset has 4 elements
    // So at max it can have 3+2+1 = 6 relations
    // However, 3 are given:
    // a0 <= a1, a0 <= a2, a1 <= a3
    RV poset_data{{0, 1}, {0, 2}, {1, 3}};
    Poset poset(4, poset_data);
    
    // We first initialize the order polytope from the poset
    OrderPolytope<Pnt> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();

    // We do not need to explicitly normalize the Polytope's matrix
    // it is handled inside the function itself
    std::cout << "compute reflection of an incident ray with the facet number 2d (the first relation facet)" << std::endl;
    Pnt ray = Point::all_ones(OP.dimension());
    ray.set_coord(0, 1.5);
    
    Pnt expected_reflected_ray = Point::all_ones(OP.dimension());
    expected_reflected_ray.set_coord(1, 1.5);

    OP.compute_reflection(ray, Point(), 2*OP.dimension());
    CHECK( (expected_reflected_ray == ray) );
}


template <typename NT>
void call_test_line_intersect() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Pnt;
    typedef typename OrderPolytope<Pnt>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Here we create Poset
    // We have 3 elements
    // And no relations (as easy to verify manually)
    RV poset_data{};
    Poset poset(3, poset_data);
    
    // We first initialize order polytope from the poset
    OrderPolytope<Pnt> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();

    // Let's compute the intersection of the order polytope with ray from (0.5, 0.5, 0.5) parallel to x-axis
    Pnt start_point(OP.dimension(), std::vector<double>(OP.dimension(), 0.5));
    Pnt expected_intersection(OP.dimension(), std::vector<double>(OP.dimension(), 0.5));
    expected_intersection.set_coord(0, 1.0);

    Pnt direction = expected_intersection - start_point;
    std::pair<double, double> curr_res = OP.line_intersect(start_point, direction, true);
    Pnt intersect_point = start_point + curr_res.first * direction;

    CHECK( (intersect_point == expected_intersection) );
}

template <typename NT>
void call_test_vec_mult() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Pnt;
    typedef typename OrderPolytope<Pnt>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Again, create Poset, 4 elements, a0 <= a3, a1 <= a3, a2 <= a3
    RV poset_data{{0, 3}, {1, 3}, {2, 3}};
    Poset poset(4, poset_data);
    
    // We initialize order polytope from the poset
    OrderPolytope<Pnt> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();
    
    // Here we multiply by all 1-vector (Ax)
    VT x = Eigen::MatrixXd::Constant(d, 1, 1.0);                            // d x 1 vector
    VT expected_res_vector = -Eigen::MatrixXd::Constant(m, 1, 1.0);         // m x 1 vector
    expected_res_vector.block(d, 0, d, 1) = Eigen::MatrixXd::Constant(d, 1, 1.0);
    expected_res_vector.block(2*d, 0, m - 2*d, 1) = Eigen::MatrixXd::Zero(m - 2*d, 1);

    VT Ax = OP.vec_mult(x);
    CHECK((expected_res_vector - Ax).norm() == 0);

    // Here we multiply by all 1-vector (A^t x)
    x = Eigen::MatrixXd::Constant(m, 1, 1.0);                        // m x 1 vector
    expected_res_vector = Eigen::MatrixXd::Constant(d, 1, 1.0);      // d x 1 vector (entries = (1, 1, 1, -3))
    expected_res_vector(3, 0) = -3.0;

    VT At_x = OP.vec_mult(x, true);
    CHECK((expected_res_vector - At_x).norm() == 0);
}


template <typename NT>
void call_test_basics() {
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point Pnt;
    typedef typename OrderPolytope<Pnt>::VT VT;
    typedef typename Poset::RT RT;
    typedef typename Poset::RV RV;

    // Once again, create a Poset with 4 elements, a0 <= a1, a0 <= a2, a1 <= a3
    RV poset_data{{0, 1}, {0, 2}, {1, 3}};
    Poset poset(4, poset_data);
    CHECK(poset.num_elem() == 4);
    CHECK(poset.num_relations() == 3);

    
    // Again, we initialize order polytope from the poset
    OrderPolytope<Pnt> OP(poset);
    unsigned int d = OP.dimension(), m = OP.num_of_hyperplanes();
    CHECK(d == 4);
    CHECK(m == 2*4 + 3);


    VT expected_dist_vector = Eigen::MatrixXd::Zero(m, 1);
    expected_dist_vector.block(d, 0, d, 1) = Eigen::MatrixXd::Constant(d, 1, 1.0);
    VT ret_dists_vector = Eigen::Map< VT >(OP.get_dists(0.0).data(), m);
    CHECK( (expected_dist_vector - ret_dists_vector).norm() == 0 );
    
    CHECK(OP.is_in(Point(4, {0.0, 0.5, 1.0, 1.0})) == -1);
    CHECK(OP.is_in(Point(4, {1.0, 0.5, 1.0, 1.0})) == 0);   // a0 <= a1 violated
    CHECK(OP.is_in(Point(4, {0.5, 0.5, 0.0, 1.0})) == 0);   // a0 <= a2 violated
    CHECK(OP.is_in(Point(4, {-0.1, 0.5, 1.0, 1.0})) == 0);  // a0 >= 0 violated
    CHECK(OP.is_in(Point(4, {1.0, 0.5, 1.0, 1.1})) == 0);   // a3 <= 1 violated
}


TEST_CASE("basics") {
    call_test_basics<double>();
}

TEST_CASE("line_intersect") {
    call_test_line_intersect<double>();
}

TEST_CASE("reflection") {
    call_test_reflection<double>();
}

TEST_CASE("vec_mult") {
    call_test_vec_mult<double>();
}


/*

[doctest] doctest version is "1.2.9"
[doctest] run with "--help" for options
compute reflection of an incident ray with the facet number 2d (the first relation facet)
===============================================================================
[doctest] test cases:      4 |      4 passed |      0 failed |      0 skipped
[doctest] assertions:     14 |     14 passed |      0 failed |
[doctest] Status: SUCCESS!


*/

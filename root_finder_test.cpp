#ifndef DISABLE_NLP_ORACLES

#include <cmath>
#include <cstdlib>
#include "doctest.h"
#include "Eigen/Eigen"
#include <gmp.h>
#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include <mps/mps.h>
#include <functional>
#include <vector>
#include <unistd.h>
#include <string>
#include <typeinfo>


#include "root_finders.hpp"

template<typename NT>
void test_newton_raphson() {
  typedef std::function<NT(NT)> func;

  func f = [](NT t) {
    return t * t - 2 * t + 1;
  };

  func grad_f = [](NT t) {
    return 2 * t - 2;
  };

  NT t0 = 5.0;

  NT t = newton_raphson<NT, func>(t0, f, grad_f, 0.001).first;

  std::cout << t << std::endl;

  CHECK(std::abs(t - 1) < 0.001);

}

template<typename NT>
void test_mpsolve() {
  NT S = NT(5);
  NT P = NT(6);

  CHECK(S >= NT(0));
  CHECK(P > NT(0));

  std::vector<NT> coeffs{P, -S, NT(1)};

  std::vector<std::pair<NT, NT>> results = mpsolve<NT>(coeffs, true);

  NT x1 = results[0].first;
  NT x2 = results[1].first;

  NT S_ = x1 + x2;
  NT P_ = x1 * x2;

  CHECK(std::abs(S_ - S) / S < 0.001);

  CHECK(std::abs(P_ - P) / P < 0.001);

}

template<typename NT>
void call_test_root_finders() {
  std::cout << "--- Testing Newton-Raphson" << std::endl;
  test_newton_raphson<NT>();

  std::cout << "--- Test mpsolve" << std::endl;
  test_mpsolve<NT>();

}

TEST_CASE("root_finders") {
  call_test_root_finders<double>();
}

#endif

/*

[doctest] doctest version is "1.2.9"
[doctest] run with "--help" for options
===============================================================================
[doctest] test cases:      0 |      0 passed |      0 failed |      0 skipped
[doctest] assertions:      0 |      0 passed |      0 failed |
[doctest] Status: SUCCESS!

*/

/**
 * @file GeneralizedEigenvalue.h
 *
 * @author Gabriel Henrique de Souza (ghdesouza@gmail.com)
 *
 * @date january 5, 2019
 *
 * @copyright Distributed under the Mozilla Public License 2.0 ( https://opensource.org/licenses/MPL-2.0 )
 *
 * @see https://github.com/ghdesouza/de_spatial_filter_for_bci
 *
 * Created on: january 5, 2019
 *
 */

#include <Eigen/Eigenvalues>
#include <iostream>

using Eigen::MatrixXf;
using Eigen::GeneralizedEigenSolver;
using namespace std;

void GeneralizedEigenvalue(double **A, double **B, double **eigenvectors, unsigned int dimensao);


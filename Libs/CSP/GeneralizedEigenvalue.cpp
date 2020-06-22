/**
 * @file GeneralizedEigenvalue.cpp
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

void GeneralizedEigenvalue(double **A, double **B, double **eigenvectors, unsigned int dimensao){

	GeneralizedEigenSolver<MatrixXf> ges;
	MatrixXf A1(dimensao,dimensao);
	MatrixXf B1(dimensao,dimensao);
	
	for(unsigned int i = 0; i < dimensao; i++){
		for(unsigned int j = 0; j < dimensao; j++){
			A1(i,j) = (float) A[i][j];
			B1(i,j) = (float) B[i][j];
		}
	}
	
	ges.compute(A1, B1);
	
	for(unsigned int i = 0; i < dimensao; i++){
		for(unsigned int j = 0; j < dimensao; j++){
			eigenvectors[j][i] = ges.eigenvectors().col(i)[j].real();
		}
	}
	
    return;
}

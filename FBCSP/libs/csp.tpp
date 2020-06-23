#ifndef CSP_H
#define CSP_H

#include "eigen/Eigen/Dense"

using Eigen::MatrixXf;
using Eigen::GeneralizedEigenSolver;
using namespace std;

template<typename T>
void GeneralizedEigenvalue(T **A, T **B, T **eigenvectors, unsigned int dimension){

	GeneralizedEigenSolver<MatrixXf> ges;
	MatrixXf A1(dimension, dimension);
	MatrixXf B1(dimension, dimension);
	
	for(unsigned int i = 0; i < dimension; i++){
		for(unsigned int j = 0; j < dimension; j++){
			A1(i,j) = (float) A[i][j];
			B1(i,j) = (float) B[i][j];
		}
	}
	
	ges.compute(A1, B1);
	
	for(unsigned int i = 0; i < dimension; i++){
		for(unsigned int j = 0; j < dimension; j++){
			eigenvectors[j][i] = ges.eigenvectors().col(i)[j].real();
		}
	}
	
    return;
}

template<typename T>
void calculate_covariance(float*** data, int* labels, int data_size, int amount_electrodes, int amount_time, T ***covariance){

    int class_temp;
    int amount_trials[2] = {0, 0};

    for(int k = 0; k < data_size; k++) amount_trials[labels[k]-1]++;

    for(int i = 0; i < amount_electrodes; i++){
        for(int j = 0; j < amount_electrodes; j++){
            covariance[0][i][j] = 0.0;
            covariance[1][i][j] = 0.0;

			for(int k = 0; k < data_size; k++){
				class_temp = labels[k]-1;
				for(int l = 0; l < amount_time; l++){
					covariance[class_temp][i][j] += data[k][i][l]*data[k][j][l];
				}
			}
			covariance[0][i][j] /= amount_trials[0];
			covariance[1][i][j] /= amount_trials[1];
		}
	}
}

void csp_spatial_filter(float*** data, int* labels, int data_size, int amount_spatial_filters, int amount_electrodes, int amount_time, float** spatial_filter){
	
    float** temp_sf = new float*[amount_electrodes];
	for(int e = 0; e < amount_electrodes; e++) temp_sf[e] = new float[amount_electrodes];
	
	float ***covariance;
	covariance = new float**[2];
	for(int i = 0; i < 2; i++){
		covariance[i] = new float*[amount_electrodes];
		for(int j = 0; j < amount_electrodes; j++){
			covariance[i][j] = new float[amount_electrodes];
		}
	}

    calculate_covariance(data, labels, data_size, amount_electrodes, amount_time,  covariance);    
    for(int j = 0; j < amount_electrodes; j++) for(int k = 0; k < amount_electrodes; k++){
        covariance[1][j][k] += covariance[0][j][k];
    }
    
    GeneralizedEigenvalue(covariance[0], covariance[1], temp_sf, amount_electrodes);
    
    int filter_temp;
    for(int n = 0; n < amount_spatial_filters; n++){
        if(n%2 == 0) filter_temp = ((int)(n/2));
        else filter_temp = (amount_electrodes-1)-((int)(n/2));
        for(int e = 0; e < amount_electrodes; e++){
            spatial_filter[n][e] = (float) temp_sf[e][filter_temp];
        }
    }

    for(int i = 0; i < 2; i++){
        for(int k = 0; k < amount_electrodes; k++){
            delete[] covariance[i][k];
        }   delete[] covariance[i];
    }       delete[] covariance;
	
	for(int e = 0; e < amount_electrodes; e++){
		delete[] temp_sf[e];
	}	delete[] temp_sf;

}

#endif // CSP_H


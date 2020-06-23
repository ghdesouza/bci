/**
 * @file naive_bayes_parzen_window.tpp
 *
 * @brief Naive Bayes Parzen Window Classifier
 *
 * @details This file implement a Naive Bayes Parzen Window Classifier
 *
 * @author Gabriel Henrique de Souza (ghdesouza@gmail.com)
 *
 * @date April 25, 2020
 *
 * @copyright Distributed under the Mozilla Public License 2.0 ( https://opensource.org/licenses/MPL-2.0 )
 *
 * @see https://github.com/ghdesouza/helpful_templates
 *
 * Created on: february 13, 2019
 *
 */

#ifndef NAIVE_BAYES_PARZEN_WINDOW_TPP
#define NAIVE_BAYES_PARZEN_WINDOW_TPP

#include <math.h>
#include "classifiers.tpp"

using namespace std;

template<typename T = float>
class Naive_Bayes_Parzen_Window : public Classifiers<T>{
	
	protected:
		int amount_trials;
		int* amount_trials_per_label;
		T*** X; // [label][dimension][trial]
		int dimension;
		int amount_labels;
		T* fitness;
	
		T** h_opt_2; // [label][dimension]

		T smoothing_kernel(T x, T h);
		T calculate_px_y(int posicao, T valor, int classe); // P( X_j | y )
		T calculate_pX_y(T *X, int y); // P( X | y )
		void calculate_py_X(T *X); // P( y | X )

	public:
		Naive_Bayes_Parzen_Window(int dimension, int amount_labels);
		~Naive_Bayes_Parzen_Window();
		
		void fit(T** data, int* labels, int data_size);
		int predict(T *X);
		T get_fitness(int label){return this->fitness[label-1];}
};

template<typename T>
Naive_Bayes_Parzen_Window<T>::Naive_Bayes_Parzen_Window(int dimension, int amount_labels){
	
	this->amount_trials = 0;
	this->dimension = dimension;
	this->amount_labels = amount_labels;
	this->amount_trials_per_label = new int[this->amount_labels];
	
	this->fitness = new T[this->amount_labels];
	for(int i = 0; i < this->amount_labels; i++) this->fitness[i] = 0.0;
	
	this->h_opt_2 = new T*[this->amount_labels];
	for(int k = 0; k < this->amount_labels; k++){
		this->h_opt_2[k] = new T[this->dimension];
	}

}

template<typename T>
Naive_Bayes_Parzen_Window<T>::~Naive_Bayes_Parzen_Window(){
	
	if(this->amount_trials != 0){
		for(int k = 0; k < this->amount_labels; k++){
			for(int j = 0; j < this->dimension; j++){
				delete[] this->X[k][j];
			}	delete[] this->X[k];
		}		delete[] this->X;
	}
	
	delete[] this->amount_trials_per_label;
	delete[] this->fitness;
	
	for(int k = 0; k < this->amount_labels; k++){
		delete[] this->h_opt_2[k];
	}	delete[] this->h_opt_2;
	
}

template<typename T>
T Naive_Bayes_Parzen_Window<T>::smoothing_kernel(T x, T h_2){
	return (1.0/sqrt(2.0*M_PI))*exp(-x*x/(2.0*h_2));
}

template<typename T>
T Naive_Bayes_Parzen_Window<T>::calculate_px_y(int position, T value, int label){
  T sum = 0;
	for(int i = 0; i < this->amount_trials_per_label[label-1]; i++){
		sum += this->smoothing_kernel(value-this->X[label-1][position][i], this->h_opt_2[label-1][position]);
	}
	return sum/this->amount_trials_per_label[label-1];
}

template<typename T>
T Naive_Bayes_Parzen_Window<T>::calculate_pX_y(T *X, int y){
	
	T pX_y = 1;

	for(int i = 0; i < this->dimension; i++){
		pX_y *= this->calculate_px_y(i, X[i], y);
	}
	
	return pX_y;
}

template<typename T>
void Naive_Bayes_Parzen_Window<T>::calculate_py_X(T *X){

	T sum = 0;
	
	for(int i = 0; i < this->amount_labels; i++){
		this->fitness[i] = this->amount_trials_per_label[i]*this->calculate_pX_y(X, i+1)/this->amount_trials;
		sum += this->fitness[i];
	}
	if(sum > 0) for(int i = 0; i < this->amount_labels; i++) this->fitness[i] /= sum;
    else for(int i = 0; i < this->amount_labels; i++) this->fitness[i] = 1/this->amount_labels;
	
	return;
}

template<typename T>
void Naive_Bayes_Parzen_Window<T>::fit(T** data, int* labels, int data_size){

	if(this->amount_trials != 0){
		for(int k = 0; k < this->amount_labels; k++){
			for(int j = 0; j < this->dimension; j++){
				delete[] this->X[k][j];
			}	delete[] this->X[k];
		}		delete[] this->X;
	}

	this->amount_trials = data_size;
	for(int i = 0; i < this->amount_labels; i++) this->amount_trials_per_label[i] = 0;
	for(int i = 0; i < this->amount_trials; i++) this->amount_trials_per_label[labels[i]-1]++;
	
	this->X = new T**[this->amount_labels];
	for(int k = 0; k < this->amount_labels; k++){
		this->X[k] = new T*[this->dimension];
		for(int j = 0; j < this->dimension; j++){
			this->X[k][j] = new T[this->amount_trials_per_label[k]];
		}
	}

	int amount_trials_temp;	
	for(int k = 0; k < this->amount_labels; k++){
		amount_trials_temp = 0;
		for(int i = 0; i < amount_trials; i++){
			if(k+1 == labels[i]){
				for(int j = 0; j < this->dimension; j++){
					this->X[k][j][amount_trials_temp] = data[i][j];
				}
				amount_trials_temp++;
			}
		}
	}

	T temp_T;
	for(int k = 0; k < this->amount_labels; k++){
		temp_T = pow(4.0/(3.0*this->amount_trials_per_label[k]), 0.4);
		for(int j = 0; j < this->dimension; j++){
			this->h_opt_2[k][j] = temp_T*variance(this->X[k][j], this->amount_trials_per_label[k]);
			if(this->h_opt_2[k][j] < 1e-8){
				this->h_opt_2[k][j] = RAND_MAX;
                #ifdef NB_LOG
				printf("\n\nERROR: variance of %dth dimension of %dth label is very small!\n\n", j, k);
                #endif
			}
		}
	}
}

template<typename T>
int Naive_Bayes_Parzen_Window<T>::predict(T *X){
	
	this->calculate_py_X(X);
	
	int label = 0;
	T prob = this->fitness[0];
	for(int i = 1; i < this->amount_labels; i++){
		if(this->fitness[i] > prob){
			label = i;
			prob = this->fitness[i];
		}
	}
	
	return (label+1);
}


#endif // NAIVE_BAYES_PARZEN_WINDOW_TPP


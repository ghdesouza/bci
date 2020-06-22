/**
 * @file mibif.tpp
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

#ifndef MIBIF_TPP
#define MIBIF_TPP

#include <math.h>
#include <stdio.h>

#include "basic.tpp"

template<typename T>
class MIBIF{
	
	private:
		int dimension;
		int amount_labels;
		int* order;
	
		T p_Xij_w(T value, T* vector, int vector_size, T h);
		T gaussian_kernel(T y, T h);
	
	public:
		
		MIBIF(int dimension, int amount_labels);
		~MIBIF();
		
		int get_dimension(){return this->dimension;}
		int get_amount_labels(){return this->amount_labels;}
		int get_id_variable(int position){return this->order[position];}
		int get_new_position(int variable){ int i = 0; for(; i < this->dimension && this->order[i] != variable; i++); if(i == this->dimension) return -1; return i;}
		void set_forced_position(int variable, int new_position);
		
		void fit(T** dataset, int* labels, int amount_trials);
		void transform(T* trial);
		void transform(T** trials, int amount_trials);
	
};

template<typename T>
MIBIF<T>::MIBIF(int dimension, int amount_labels){

	this->dimension = dimension;
	this->amount_labels = amount_labels;

	this->order = new int[this->dimension];
	for(int i = 0; i < this->dimension; i++) this->order[i] = i;
}

template<typename T>
MIBIF<T>::~MIBIF(){

	delete[] this->order;
}

template<typename T>
T MIBIF<T>::gaussian_kernel(T y, T h){
	return 1.0/sqrt(2.0*M_PI)*exp(-y*y/(2.0*h*h));
}

template<typename T>
void MIBIF<T>::fit(T** dataset, int* labels, int amount_trials){

	T* I = new T[this->dimension]; // vector with value proportional at the mutual information
	for(int j = 0; j < this->dimension; j++) I[j] = 0.0;
	
	int* cont_w = new int[this->amount_labels]; // amount of trials at each labels
	for(int w = 0; w < this->amount_labels; w++) cont_w[w] = cont(w+1, labels, amount_trials);
	
	T** p_w_Xij = new T*[this->amount_labels]; // probability of the Xij value to be w class
	for(int w = 0; w < this->amount_labels; w++) p_w_Xij[w] = new T[amount_trials];
	
	T* vec_j = new T[amount_trials]; // vector with all values of the jth dimension
	T h; // std for gaussian kernel
	
	T temp_T;
	int temp_int;
	
	for(int j = 0; j < this->dimension; j++){
		
		for(int w = 0; w < this->amount_labels; w++){
			
			// select all values of jth dimension of wth class
			temp_int = 0;
			for(int i = 0; i < amount_trials; i++){
				if(labels[i] == (w+1)){
					vec_j[temp_int] = dataset[i][j];
					temp_int++;
				}
			}
			
			// calculate std for gaussian kernel
			h = sqrt(variance(vec_j, cont_w[w]))*pow((4.0/(3.0*cont_w[w])),0.2);
			if(h < 1e-8){
				printf("\n\nERROR: variance of %dth variable in %dth class equal zero!\n\n", j, w+1);
				return;
			}
			
			// bayes theorem without normalization
			for(int i = 0; i < amount_trials; i++){
				p_w_Xij[w][i] = p_Xij_w(dataset[i][j], vec_j, cont_w[w], h)*((T) cont_w[w])/amount_trials;
				//printf("%f\n", p_Xij_w(dataset[i][j], vec_j, cont_w[w], h));
			}
		}
		
		for(int i = 0; i < amount_trials; i++){
			
			// normalization of bayes theorem
			temp_T = 0.0;
			for(int w = 0; w < this->amount_labels; w++) temp_T += p_w_Xij[w][i];
			for(int w = 0; w < this->amount_labels; w++) p_w_Xij[w][i] /= temp_T;
			
			// calculate mutual information without H(w) term
			for(int w = 0; w < this->amount_labels; w++){
				I[j] += p_w_Xij[w][i]*log2(p_w_Xij[w][i]);
			}
		}
	}

	// ordenation order by mutual information
	// insertion sort is a good idea because in the most problems the dimension isn't very large (>500)
	T max_value;
	int pos_max;
	for(int o = 0; o < this->dimension; o++) order[o] = o;
	for(int i = 0; i < this->dimension; i++){
		pos_max = i;
		max_value = I[i];
		for(int j = i+1; j < this->dimension; j++){
			if(I[j] > max_value){
				pos_max = j;
				max_value = I[j];
			}
		}
		
		// swaps
		temp_T = I[i];
		I[i] = I[pos_max];
		I[pos_max] = temp_T;
		temp_int = this->order[i];
		this->order[i] = this->order[pos_max];
		this->order[pos_max] = temp_int;
		
	}

	// delete structs created for find the order
	delete[] cont_w;
	for(int w = 0; w < this->amount_labels; w++){
		delete[] p_w_Xij[w];
	}	delete[] p_w_Xij;
	delete[] vec_j;	
	delete[] I;
	
	
	return;
}

template<typename T>
T MIBIF<T>::p_Xij_w(T value, T* vector, int vector_size, T h){
	T exit = 0;
	for(int i = 0; i < vector_size; i++){
		exit += this->gaussian_kernel(value-vector[i], h);
	}
	return (exit/vector_size);
}

template<typename T>
void MIBIF<T>::transform(T* trial){
	
	T* temp = new T[this->dimension];
	
	for(int j = 0; j < this->dimension; j++) temp[j] = trial[this->order[j]];
	for(int j = 0; j < this->dimension; j++) trial[j] = temp[j];
	
	delete[] temp;
	
	return;
	
}

template<typename T>
void MIBIF<T>::transform(T** trials, int amount_trials){
	
	T* temp = new T[this->dimension];
	
	for(int t = 0; t < amount_trials; t++){
		for(int j = 0; j < this->dimension; j++) temp[j] = trials[t][this->order[j]];
		for(int j = 0; j < this->dimension; j++) trials[t][j] = temp[j];
	}

	delete[] temp;
	
	return;
	
}

template<typename T>
void MIBIF<T>::set_forced_position(int variable, int new_position){
	
	int past_position = this->get_new_position(variable);
	int temp_int;
	
	if(new_position < past_position){
		for(; new_position < past_position; past_position--){
			temp_int = this->order[past_position];
			this->order[past_position] = this->order[past_position-1];
			this->order[past_position-1] = temp_int;
		}
	}else{
		for(; new_position > past_position; past_position++){
			temp_int = this->order[past_position];
			this->order[past_position] = this->order[past_position+1];
			this->order[past_position+1] = temp_int;
		}
	}
	
}

#endif //MIBIF_TPP


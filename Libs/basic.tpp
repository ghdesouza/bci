/**
 * @file basic.tpp
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
 
#ifndef BASIC_CPP
#define BASIC_CPP

template<typename T>
T mean(T* vector, int vector_size){
	
	T value = 0;

	for(int i = 0; i < vector_size; i++){
		value += vector[i];
	}
	
	return (value/vector_size);
}

template<typename T>
T variance(T* vector, int vector_size){

	T mean_value = mean(vector, vector_size);
	T var = 0;
	T temp;
	
	for(int i = 0; i < vector_size; i++){
		temp = (vector[i]-mean_value);
		var += temp*temp;
	}
	
	return (var/vector_size);
}

template<typename T>
int cont(T value, T* vector, int vector_size){
	
	int sum = 0;

	for(int i = 0; i < vector_size; i++){
		if(vector[i] == value) sum++;
	}
	
	return sum;
}

#endif //BASIC_CPP


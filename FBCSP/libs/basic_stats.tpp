/**
 * @file basic_stats.tpp
 *
 * @brief Basics functions for statistic
 *
 * @details This file implement basics functions in statistic
 *
 * @author Gabriel Henrique de Souza (ghdesouza@gmail.com)
 *
 * @date february 13, 2019
 *
 * @copyright Distributed under the Mozilla Public License 2.0 ( https://opensource.org/licenses/MPL-2.0 )
 *
 * @see https://github.com/ghdesouza/helpful_templates
 *
 * Created on: february 13, 2019
 *
 */

#ifndef BASIC_STATS_TPP
#define BASIC_STATS_TPP

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
	
	return (var/(vector_size-1));
}

template<typename T>
int cont(T value, T* vector, int vector_size){
	
	int sum = 0;

	for(int i = 0; i < vector_size; i++){
		if(vector[i] == value) sum++;
	}
	
	return sum;
}

#endif // BASIC_STATS_TPP


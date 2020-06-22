/**
 * @file classifier.tpp
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

#ifndef CLASSIFIERS_TPP
#define CLASSIFIERS_TPP

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

template<typename T = float>
class Classifiers{
	
    protected:
		int dimension;
		int amount_labels;
		
    public:
		virtual ~Classifiers(){};
		
		virtual int get_dimension(){return this->dimension;}
		virtual int get_amount_labels(){return this->amount_labels;}
		virtual T get_fitness(int label) = 0;
		
		virtual void fit(T **X, int *Y, int amount_trials) = 0;
		virtual int predict(T *X) = 0;
		
};

#endif // CLASSIFIERS_TPP


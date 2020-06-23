/**
 * @file dataset.tpp
 *
 * @brief Abstract class for general classifiers
 *
 * @details This file implement a abstract class with avaliations approach.
 *
 * @author Gabriel Henrique de Souza (ghdesouza@gmail.com)
 *
 * @date november 01, 2019
 *
 * @copyright Distributed under the Mozilla Public License 2.0 ( https://opensource.org/licenses/MPL-2.0 )
 *
 * @see https://github.com/ghdesouza/helpful_templates
 *
 * Created on: february 13, 2019
 *
 */

#ifndef CLASSIFIERS_TPP
#define CLASSIFIERS_TPP

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "basic_stats.tpp"

using namespace std;

typedef struct metrics{
    float accurace;
    float kappa;
    float mse;
    float crossentropy;
    float auc;
} metrics;

template<typename T = float>
class Classifiers{
	
    protected:
		int dimension;
		int amount_labels;
		
    public:
		virtual ~Classifiers(){};
		
		virtual int get_dimension(){return this->dimension;}
		virtual int get_amount_labels(){return this->amount_labels;}
		
		virtual void fit(T** data, int* labels, int data_size) = 0;
		virtual int predict(T *X) = 0;
		virtual T get_fitness(int label) = 0;
		
		virtual float kappa(T** data, int* labels, int data_size);
		virtual float accurace(T** data, int* labels, int data_size);
		virtual float cross_entropy(T** data, int* labels, int data_size);
		virtual float mean_squared_error(T** data, int* labels, int data_size);
		virtual float auc(T** data, int* labels, int data_size);
        
        virtual metrics full_metrics(T** data, int* labels, int data_size);
};


template<typename T>
float Classifiers<T>::kappa(T** data, int* labels, int data_size){
    int** confusion_matrix = new int*[this->amount_labels];
    for(int i = 0; i < this->amount_labels; i++){
        confusion_matrix[i] = new int[this->amount_labels];
        for(int j = 0; j < this->amount_labels; j++){
            confusion_matrix[i][j] = 0;
        }
    }
    int amount_trials;
    amount_trials = data_size;
    for(int i = 0; i < amount_trials; i++){
        confusion_matrix[labels[i]-1][this->predict(data[i])-1]++;
    }
    float p_0, p_e = 0.0, total = 0.0, correct = 0.0;
    float *expected = new float[this->amount_labels];
    float *finded = new float[this->amount_labels];
    for(int i = 0; i < this->amount_labels; i++){
        expected[i] = 0;
        finded[i] = 0;
        correct += confusion_matrix[i][i];
        for(int j = 0; j < this->amount_labels; j++){
            expected[i] += confusion_matrix[j][i];
            finded[i] += confusion_matrix[i][j];
            total += confusion_matrix[i][j];
        }
    }

    p_0 = correct/total;
    for(int i = 0; i < this->amount_labels; i++) p_e += (finded[i]/total)*(expected[i]/total);

    delete[] expected;
    delete[] finded;
    
    for(int i = 0; i < this->amount_labels; i++){
        delete[] confusion_matrix[i];
    }	delete[] confusion_matrix;
    
    return (p_0-p_e)/(1.0-p_e);
    
}

template<typename T>
float Classifiers<T>::accurace(T** data, int* labels, int data_size){
    float correct = 0.0;
    for(int i = 0; i < data_size; i++){
        if(this->predict(data[i]) == labels[i]){
            correct += 1;
        }
    }
    
    return correct/data_size;
}

template<typename T>
float Classifiers<T>::cross_entropy(T** data, int* labels, int data_size){
    T crossentropy = 0.0;
    int temp_class;
    T temp_val;

    for(int i = 0; i < data_size; i++){
        
        temp_class = this->predict(data[i]);
        temp_val = this->get_fitness(labels[i]);
        if(temp_val < 0.01) temp_val = 0.01;
        crossentropy -= log(temp_val);
    }
    return crossentropy/data_size;
}

template<typename T>
float Classifiers<T>::mean_squared_error(T** data, int* labels, int data_size){
    T error = 0;
    int temp_class;

    for(int i = 0; i < data_size; i++){
        temp_class = this->predict(data[i]);
        error += pow((1-this->get_fitness(labels[i])), 2);
        for(int j = 1; j <= this->amount_labels; j++) if(j != labels[i]) error += pow(this->get_fitness(j), 2);
    }
    return error/(data_size*this->amount_labels);
}

template<typename T>
float Classifiers<T>::auc(T** data, int* labels, int data_size){
    
    float auc = 0;
    int len_pos, len_neg;
    float temp_auc;
    float** probs = new float*[this->amount_labels];
    int* trial_per_class = new int[this->amount_labels];
    float** probs_acc = new float*[2];
    for(int i = 0; i < 2; i++) probs_acc[i] = new float[data_size];
    for(int i = 0; i < this->amount_labels; i++){
        probs[i] = new float[data_size];
        trial_per_class[i] = 0;
    }
    
    for(int i = 0; i < data_size; i++){
        trial_per_class[this->predict(data[i])-1]++;
        for(int j = 0; j < this->amount_labels; j++) probs[j][i] = this->get_fitness(j+1);
    }
    
    for(int l = 1; l <= this->amount_labels; l++){
        temp_auc = 0;
        len_pos = len_neg = 0;
        for(int i = 0; i < data_size; i++){
            if(labels[i] == l){
                probs_acc[0][len_pos] = probs[l-1][i];
                len_pos++;
            }else{
                probs_acc[1][len_neg] = probs[l-1][i];
                len_neg++;
            }
        }
        for(int p = 0; p < len_pos; p++){
            for(int n = 0; n < len_neg; n++){
                if(probs_acc[0][p] > probs_acc[1][n]) temp_auc+=1.0;
                else if(probs_acc[0][p] == probs_acc[1][n]) temp_auc+=0.5;
            }
        }
        auc += temp_auc/(len_pos*len_neg);
    }
    
    for(int i = 0; i < this->amount_labels; i++) delete[] probs[i];
    for(int i = 0; i < 2; i++) delete[] probs_acc[i];
    delete[] probs;
    delete[] probs_acc;
    delete[] trial_per_class;
    
    return auc/this->amount_labels;
}

template<typename T>
metrics Classifiers<T>::full_metrics(T** data, int* labels, int data_size){
    
	float kappa, mean_squared_error, crossentropy, accurace, auc;
	kappa = 0; mean_squared_error = 0; crossentropy = 0; accurace = 0; auc = 0;
	int temp_class;
	T temp_val;
    
    int len_pos, len_neg;
    float temp_auc;
    float** probs_acc = new float*[2];
    for(int i = 0; i < 2; i++) probs_acc[i] = new float[data_size];
    float** probs = new float*[this->amount_labels];
    int* trial_per_class = new int[this->amount_labels];
	int** confusion_matrix = new int*[this->amount_labels];
	for(int i = 0; i < this->amount_labels; i++){
        probs[i] = new float[data_size];
        trial_per_class[i] = 0;
		confusion_matrix[i] = new int[this->amount_labels];
		for(int j = 0; j < this->amount_labels; j++) confusion_matrix[i][j] = 0;
	}
	for(int i = 0; i < data_size; i++){
		temp_class = this->predict(data[i]);
        
        trial_per_class[temp_class-1]++;
        for(int j = 0; j < this->amount_labels; j++) probs[j][i] = this->get_fitness(j+1);
        
		confusion_matrix[labels[i]-1][temp_class-1]++;
        
        temp_val = this->get_fitness(labels[i]);
        if(temp_val < 0.01) temp_val = 0.01;
        crossentropy -= log(temp_val);
        
        mean_squared_error += pow((1-temp_val), 2);
        for(int j = 1; j <= this->amount_labels; j++) if(j != labels[i]) mean_squared_error += pow(this->get_fitness(j), 2);
	}
	
    for(int l = 1; l <= this->amount_labels; l++){
        temp_auc = 0;
        len_pos = len_neg = 0;
        for(int i = 0; i < data_size; i++){
            if(labels[i] == l){
                probs_acc[0][len_pos] = probs[l-1][i];
                len_pos++;
            }else{
                probs_acc[1][len_neg] = probs[l-1][i];
                len_neg++;
            }
        }
        for(int p = 0; p < len_pos; p++){
            for(int n = 0; n < len_neg; n++){
                if(probs_acc[0][p] > probs_acc[1][n]) temp_auc+=1.0;
                else if(probs_acc[0][p] == probs_acc[1][n]) temp_auc+=0.5;
            }
        }
        auc += temp_auc/(len_pos*len_neg);
    }
	
	float p_0, p_e = 0.0, total = 0.0, correct = 0.0;
	float *expected = new float[this->amount_labels];
	float *finded = new float[this->amount_labels];
	for(int i = 0; i < this->amount_labels; i++){
		expected[i] = 0;
		finded[i] = 0;
		correct += confusion_matrix[i][i];
		for(int j = 0; j < this->amount_labels; j++){
			expected[i] += confusion_matrix[j][i];
			finded[i] += confusion_matrix[i][j];
			total += confusion_matrix[i][j];
		}
	}

	p_0 = correct/total;
	for(int i = 0; i < this->amount_labels; i++) p_e += (finded[i]/total)*(expected[i]/total);

	delete[] expected;
	delete[] finded;
	
	for(int i = 0; i < this->amount_labels; i++){
		delete[] confusion_matrix[i];
	}	delete[] confusion_matrix;
    
    for(int i = 0; i < this->amount_labels; i++) delete[] probs[i];
    for(int i = 0; i < 2; i++) delete[] probs_acc[i];
    delete[] probs;
    delete[] probs_acc;
    delete[] trial_per_class;
    
	accurace = p_0;
	kappa = (p_0-p_e)/(1.0-p_e);
	crossentropy = crossentropy/data_size;
	mean_squared_error = mean_squared_error/(data_size*this->amount_labels);
    auc = auc/this->amount_labels;
    
    metrics metric;
    metric.kappa = kappa;
    metric.accurace = accurace;
    metric.mse = mean_squared_error; // Brier Score
    metric.crossentropy = crossentropy;
    metric.auc = auc;
    
    return metric;
}

#endif // CLASSIFIERS_TPP


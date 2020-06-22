/**
 * @file main.cpp
 *
 * @brief main DE Applied for Spatial Filters Optimization for BCI
 *
 * @details This file contain the main function of DE Applied for Spatial Filters Optimization for BCI
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

#include <fstream>

// if CSP_LIB defined CSP is used and if CSP_LIB not defined is used DE
#define CSP_LIB 1

// if OVR defined one-versus-all is used
#define OVR 1


#ifdef CSP_LIB
	#include "Libs/CSP/GeneralizedEigenvalue.h"
	template<typename T>
	void fit_spatial_filter(T***** X, T*** spatial_filters);
	template<typename T>
	void fit_spatial_filter_ovr(T***** X, T**** spatial_filters);
#endif

#include "Libs/bci_ovr.tpp"
#include "Libs/DE.hpp"

#include "Setting/basic_setting.h"
#include "Setting/config_set.h"

int seed = 1;

template<typename T>
void load_dataset(string name_file, int amount_electrodes, int amount_time, int amount_trials, T*** data);

void fit_de_spatial_filter();

template<typename T>
void mix(T***** data, int trials_per_label);

#ifdef OVR
	BCI_OVR<float>* bci;
#else
	BCI<float>* bci;
#endif

int id_suject;
string name_train_file = "train_exit.txt";
string name_test_file =   "test_exit.txt";

int main(){
    
	#ifdef OVR
		bci = new BCI_OVR<float>(amount_electrodes, amount_classifier_time, amount_band_filters, amount_spatial_filters_per_band, selected_features, amount_labels);
	#else
		bci = new BCI<float>(amount_electrodes, amount_classifier_time, amount_band_filters, amount_spatial_filters_per_band, selected_features, amount_labels);
	#endif

	srand(seed);
	for(id_suject = 1; id_suject <= 9; id_suject++){
		
		amount_test_per_label = (int)(amount_trials_per_label[id_suject-1]/kfold_size);
		amount_train_per_label = (int)(amount_trials_per_label[id_suject-1]-amount_test_per_label);
		{ // loading dataset
			data = new float***[amount_labels];
			for(int l = 0; l < amount_labels; l++){
				data[l] = new float**[amount_trials_per_label[id_suject-1]];
				for(int i = 0; i < amount_trials_per_label[id_suject-1]; i++){
					data[l][i] = new float*[amount_electrodes];
					for(int e = 0; e < amount_electrodes; e++){
						data[l][i][e] = new float[amount_full_time];
					}
				}
			}
			for(int l = 0; l < amount_labels; l++){
				load_dataset(names_datasets[l][(id_suject-1)], amount_electrodes, amount_full_time, amount_trials_per_label[id_suject-1], data[l]);
			}
		}
		{ // band pass filters
			data_extended = new float****[amount_labels];
			for(int l = 0; l < amount_labels; l++){
				data_extended[l] = new float***[amount_trials_per_label[id_suject-1]];
				for(int i = 0; i < amount_trials_per_label[id_suject-1]; i++){
					data_extended[l][i] = new float**[amount_band_filters];
					for(int bf = 0; bf < amount_band_filters; bf++){
						data_extended[l][i][bf] = new float*[amount_electrodes];
						for(int e = 0; e < amount_electrodes; e++){
							data_extended[l][i][bf][e] = new float[amount_full_time-start_time];
						}
					}
				}
			}
			for(int l = 0; l < amount_labels; l++){
				#pragma omp parallel for
				for(int i = 0; i < amount_trials_per_label[id_suject-1]; i++){
					bci->preprocessing(data[l][i], data_extended[l][i], band_pass, acquisition_frequency, amount_full_time, start_time);
				}
			}
		}
		{ // delete dataset
			for(int l = 0; l < amount_labels; l++){
				for(int i = 0; i < amount_trials_per_label[id_suject-1]; i++){
					for(int e = 0; e < amount_electrodes; e++){
						delete[] data[l][i][e];
					}	delete[] data[l][i];
				}		delete[] data[l];
			}			delete[] data;
		}
		for(int num_kfold = 1; num_kfold <= amount_kfold; num_kfold++){
			mix(data_extended, amount_trials_per_label[id_suject-1]);
			for(int id_fold = 1; id_fold <= kfold_size; id_fold++){
				{ // separating train and test group
					int temp_train = 0, temp_test = 0;
					data_extended_train = new float****[amount_labels];
					data_extended_test = new float****[amount_labels];
					for(int l = 0; l < amount_labels; l++){
						data_extended_train[l] = new float***[amount_train_per_label];
						data_extended_test[l] = new float***[amount_test_per_label];
						temp_train = 0;
						temp_test = 0;
						for(int i = 0; i < amount_trials_per_label[id_suject-1]; i++){
							if(i%kfold_size == id_fold-1){
								data_extended_test[l][temp_test] = data_extended[l][i];
								temp_test++;
							}else{
								data_extended_train[l][temp_train] = data_extended[l][i];
								temp_train++;
							}
						}
					}
				}
				{ // FBCSP or DE fit
					#ifdef CSP_LIB
						#ifdef OVR
							fit_spatial_filter_ovr(data_extended_train, bci->spatial_filters);
						#else
							fit_spatial_filter(data_extended_train, bci->spatial_filters);
						#endif
					#else
						fit_de_spatial_filter();
					#endif
				}
				{ // Extraction and Selection of features and classifier fit
					bci->fit_preprocess_without_spatial(data_extended_train, amount_train_per_label);
				}
				{ // Calculate kappa value
					printf("%d %d %d\n", id_suject, num_kfold, id_fold);
					
					file = fopen(name_test_file.c_str(), "app");
					if(!ZOH) fprintf(file, "\t%f", bci->kappa_calculate(data_extended_test, amount_test_per_label));
					else fprintf(file, "\t%f", bci->zero_hold(data_extended_test, amount_test_per_label, amount_full_time-start_time));
					fclose(file);
					
					file = fopen(name_train_file.c_str(), "app");
					if(!ZOH) fprintf(file, "\t%f", bci->kappa_calculate(data_extended_train, amount_train_per_label));
					else fprintf(file, "\t%f", bci->zero_hold(data_extended_train, amount_train_per_label, amount_full_time-start_time));
					fclose(file);
					
				}
				{ // delete separation between train and test separation
					for(int l = 0; l < amount_labels; l++){
						delete[] data_extended_train[l];
						delete[] data_extended_test[l];
					}	delete[] data_extended_train;
						delete[] data_extended_test;
				}
			}
			file = fopen(name_test_file.c_str(), "app");
			fprintf(file, "\n");
			fclose(file);
			file = fopen(name_train_file.c_str(), "app");
			fprintf(file, "\n");
			fclose(file);
		}
		{ // delete data_extended
			for(int l = 0; l < amount_labels; l++){
				for(int i = 0; i < amount_trials_per_label[id_suject-1]; i++){
					for(int bf = 0; bf < amount_band_filters; bf++){
						for(int e = 0; e < amount_electrodes; e++){
							delete[] data_extended[l][i][bf][e];
						}	delete[] data_extended[l][i][bf];
					}		delete[] data_extended[l][i];
				}			delete[] data_extended[l];
			}				delete[] data_extended;
		}
		file = fopen(name_test_file.c_str(), "app");
		fprintf(file, "\n");
		fclose(file);
		file = fopen(name_train_file.c_str(), "app");
		fprintf(file, "\n");
		fclose(file);
	}
	
	delete bci;
	return 0;
}

template<typename T>
void load_dataset(string name_file, int amount_electrodes, int amount_time, int amount_trials, T*** data){
	
	ifstream file(name_file.c_str());
	if(!file.is_open()) {
		cout<<"Error to find the file."<<endl;
		cin.get();
	return;
	}
	
	for(int i = 0; i < amount_trials; i++){
		for(int e = 0; e < amount_electrodes; e++){
			for(int t = 0; t < amount_time; t++){
				file >> data[i][e][t];
			}
		}
	}
	file.close();
	return;
	
}

template<typename T>
void mix(T***** data, int trials_per_label){
	
	int* order = new int[trials_per_label];
	int pos_max;
	int max_val;
	T*** temp_T;
	int temp_int;
	
	for(int l = 0; l < amount_labels; l++){
	
		for(int i = 0; i < trials_per_label; i++) order[i] = (int)rand();

		for(int i = 0; i < trials_per_label; i++){
			pos_max = i;
			max_val = order[i];
			for(int j = i+1; j < trials_per_label; j++){
				if(order[j] > max_val){
					pos_max = j;
					max_val = order[j];
				}
			}
			
			// swaps
			temp_T = data[l][i];
			data[l][i] = data[l][pos_max];
			data[l][pos_max] = temp_T;
			temp_int = order[i];
			order[i] = order[pos_max];
			order[pos_max] = temp_int;
		}
	}
	
	delete[] order;
}

float fitness_DE(float* x, int dim){
	
	int t = 0;
	#ifdef OVR
		for(int l = 0; l < amount_labels; l++){
			for(int bf = 0; bf < amount_band_filters; bf++){
				for(int n = 0; n < amount_spatial_filters_per_band; n++){
					for(int e = 0; e < amount_electrodes; e++){
						bci->spatial_filters[l][bf][n][e] = x[t];
						t++;
					}
				}
			}
		}	
	#else
		for(int bf = 0; bf < amount_band_filters; bf++){
			for(int n = 0; n < amount_spatial_filters_per_band; n++){
				for(int e = 0; e < amount_electrodes; e++){
					bci->spatial_filters[bf][n][e] = x[t];
					t++;
				}
			}
		}
	#endif
	
	bci->fit_preprocess_without_spatial(data_extended_train, amount_train_per_label);
	return bci->crossentropy_calculate(data_extended_train, amount_train_per_label, 0);
	
}

void lim_DE(float* x, int dim){

    for(int i = 0; i < dim; i++){
        if(x[i] < -1) x[i] = -1;
        else if(x[i] > 1) x[i] = 1;
    }
}

void fit_de_spatial_filter(){
	
	#ifdef OVR
	int dimension = amount_labels*amount_band_filters*amount_spatial_filters_per_band*amount_electrodes;
	#else
    int dimension = amount_band_filters*amount_spatial_filters_per_band*amount_electrodes;
	#endif
    int tam_populacao = 50;
    float escala_vetor = 0.7;
    float prob_mudar = 0.8;
    
    Evolucao_Diferencial *de = new Evolucao_Diferencial(dimension, tam_populacao, fitness_DE, escala_vetor, prob_mudar);
    de->set_restricao(lim_DE);
	
	for(int i = 0; i <= 500; i++){
        de->proxima_geracao();
        if(i % 50 == 0)
            cout << "Iteration: " << (i+1) << "  Best: " << de->get_melhor_individuo()[dimension] << endl;
    }
    
	int t = 0;
	float* x = de->get_melhor_individuo();
	#ifdef OVR
		for(int l = 0; l < amount_labels; l++){
			for(int bf = 0; bf < amount_band_filters; bf++){
				for(int n = 0; n < amount_spatial_filters_per_band; n++){
					for(int e = 0; e < amount_electrodes; e++){
						bci->spatial_filters[l][bf][n][e] = x[t];
						t++;
					}
				}
			}
		}	
	#else
		for(int bf = 0; bf < amount_band_filters; bf++){
			for(int n = 0; n < amount_spatial_filters_per_band; n++){
				for(int e = 0; e < amount_electrodes; e++){
					bci->spatial_filters[bf][n][e] = x[t];
					t++;
				}
			}
		}
	#endif
	
}

#ifdef CSP_LIB

template<typename T, typename E>
void calculate_covariance(T**** dataset, int bf, E **covariance, int amount_trials){
	for(int i = 0; i < amount_electrodes; i++){
		for(int j = 0; j < amount_electrodes; j++){
			covariance[i][j] = 0.0;
			for(int k = 0; k < amount_trials; k++){
				for(int l = 0; l < amount_classifier_time; l++){
					covariance[i][j] += (E)(dataset[k][bf][i][l]*dataset[k][bf][j][l]);
				}
			}
			covariance[i][j] /= amount_trials;
		}
	}
}

template<typename T, typename E>
void calculate_covariance_ovr(T***** dataset, int bf, E*** covariance, int label, int amount_trials){
	
	for(int i = 0; i < amount_electrodes; i++){
		for(int j = 0; j < amount_electrodes; j++){
			covariance[0][i][j] = 0.0;
			covariance[1][i][j] = 0.0;
		}
	}
	
	for(int il = 0; il < amount_labels; il++){
		for(int i = 0; i < amount_electrodes; i++){
			for(int j = 0; j < amount_electrodes; j++){
				for(int k = 0; k < amount_trials; k++){
					for(int l = 0; l < amount_classifier_time; l++){
						covariance[((int)(il!=label))][i][j] += (E)(dataset[il][k][bf][i][l]*dataset[il][k][bf][j][l]);
					}
				}
			}
		}
	}
	for(int i = 0; i < amount_electrodes; i++){
		for(int j = 0; j < amount_electrodes; j++){
			covariance[0][i][j] /= amount_trials;
			covariance[1][i][j] /= amount_trials*(amount_labels-1);
		}
	}
}

template<typename T>
void fit_spatial_filter(T***** X, T*** spatial_filters){
	
	double** temp_sf = new double*[amount_electrodes];
	for(int e = 0; e < amount_electrodes; e++) temp_sf[e] = new double[amount_electrodes];
	
	double ***covariance;
	covariance = new double**[2];
	for(int i = 0; i < 2; i++){
		covariance[i] = new double*[amount_electrodes];
		for(int j = 0; j < amount_electrodes; j++){
			covariance[i][j] = new double[amount_electrodes];
		}
	}
	
	int filter_temp;
	for(int bf = 0; bf < amount_band_filters; bf++){
		calculate_covariance(X[0], bf, covariance[0], amount_train_per_label);
		calculate_covariance(X[1], bf, covariance[1], amount_train_per_label);
		
		for(int j = 0; j < amount_electrodes; j++) for(int k = 0; k < amount_electrodes; k++){
			covariance[1][j][k] += covariance[0][j][k];
		}
		
		GeneralizedEigenvalue(covariance[0], covariance[1], temp_sf, amount_electrodes);
		
		for(int n = 0; n < amount_spatial_filters_per_band; n++){
			if(n%2 == 0) filter_temp = ((int)(n/2));
			else filter_temp = (amount_electrodes-1)-((int)(n/2));
			for(int e = 0; e < amount_electrodes; e++){
				spatial_filters[bf][n][e] = (T) temp_sf[e][filter_temp];
			}
		}
	}
	
	for(int i = 0; i < 2; i++){
		for(int k = 0; k < amount_electrodes; k++){
			delete[] covariance[i][k];
		}	delete[] covariance[i];
	}		delete[] covariance;
	
	for(int e = 0; e < amount_electrodes; e++){
		delete[] temp_sf[e];
	}	delete[] temp_sf;
	
}

template<typename T>
void fit_spatial_filter_ovr(T***** X, T**** spatial_filters){
	
	double** temp_sf = new double*[amount_electrodes];
	for(int e = 0; e < amount_electrodes; e++) temp_sf[e] = new double[amount_electrodes];
	
	double ***covariance;
	covariance = new double**[2];
	for(int i = 0; i < 2; i++){
		covariance[i] = new double*[amount_electrodes];
		for(int j = 0; j < amount_electrodes; j++){
			covariance[i][j] = new double[amount_electrodes];
		}
	}
	
	int filter_temp;
	for(int l = 0; l < amount_labels; l++){
		for(int bf = 0; bf < amount_band_filters; bf++){
			calculate_covariance_ovr(X, bf, covariance, l, amount_train_per_label);
			
			for(int j = 0; j < amount_electrodes; j++) for(int k = 0; k < amount_electrodes; k++){
				covariance[1][j][k] += covariance[0][j][k];
			}
			
			GeneralizedEigenvalue(covariance[0], covariance[1], temp_sf, amount_electrodes);
			
			for(int n = 0; n < amount_spatial_filters_per_band; n++){
				if(n%2 == 0) filter_temp = ((int)(n/2));
				else filter_temp = (amount_electrodes-1)-((int)(n/2));
				for(int e = 0; e < amount_electrodes; e++){
					spatial_filters[l][bf][n][e] = (T) temp_sf[e][filter_temp];
				}
			}
		}
	}
	
	for(int i = 0; i < 2; i++){
		for(int k = 0; k < amount_electrodes; k++){
			delete[] covariance[i][k];
		}	delete[] covariance[i];
	}		delete[] covariance;
	
	for(int e = 0; e < amount_electrodes; e++){
		delete[] temp_sf[e];
	}	delete[] temp_sf;
	
}

#endif


/**
 * @file bci.tpp
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

#ifndef BCI_H
#define BCI_H

#include "fourier_transform.tpp"
#include "mibif.tpp"
#include "Naive_Bayes_Gaussian.tpp"
#include "Naive_Bayes_Parzen_Window.tpp"

double coeficiente_kappa(int** dados, int dimensao){
    double p_0;          // porcentagem de concordancia
    double p_e = 0.0;    // probabilidade de concordancia randomica
    double total = 0.0;
    double concordaram = 0.0;
    double *quant_esperado = new double[dimensao];
    double *quant_encontrado = new double[dimensao];
    for(int i = 0; i < dimensao; i++){
        quant_encontrado[i] = 0;
        quant_esperado[i] = 0;
        concordaram += dados[i][i];
        for(int j = 0; j < dimensao; j++){
            quant_encontrado[i] += dados[i][j];
            quant_esperado[i] += dados[j][i];
            total += dados[i][j];
        }
    }
    p_0 = concordaram/total;
    for(int i = 0; i < dimensao; i++) p_e += (quant_encontrado[i]/total)*(quant_esperado[i]/total);

    delete[] quant_esperado;
    delete[] quant_encontrado;
    return (p_0-p_e)/(1.0-p_e);
}

template<typename T>
class BCI{
	
	protected:
	
		int amount_electrodes;
		int amount_time;
		int amount_band_filters;
		int amount_spatial_filters_per_band;
		int amount_selected_features;
		int amount_labels;

		MIBIF<T>* mibif;
		Classifiers<T>* classifier;

	public:
	
		T*** spatial_filters; // [band][id_filter][electrode]
	
		BCI(int amount_electrodes, int amount_time, int amount_band_filters, int amount_spatial_filters_per_band, int amount_selected_features, int amount_labels);
		virtual ~BCI();
	
		void preprocessing(T** X, T*** X_bank, T band_pass_limits[][2], T acquisition_frequency, int full_time, int start_time); // X[electrode][time], X_bank[id_filter][electrode][time]
		void spatial_filter(T*** X, T*** X_filtered);
		void shift_spatial_filter(T*** X, T*** X_filtered, int shift);
		void log_power(T*** X, T* C);
		void paired_mibif();
		void fit_preprocess_without_spatial(T***** X_bank, int amount_trial_per_label);
		int predict(T*** X_bank);
		int shift_predict(T*** X_bank, int shift);
		T zero_hold(T***** X_bank, int amount_trials_per_label, int max_position);
		T kappa_calculate(T***** X_bank, int amount_trials_per_label, int position);
		T kappa_calculate(T***** X_bank, int amount_trials_per_label){ return kappa_calculate(X_bank, amount_trials_per_label, 0); }
		T crossentropy_calculate(T***** X_bank, int amount_trials_per_label, int position);
	
};

template<typename T>
BCI<T>::BCI(int amount_electrodes, int amount_time, int amount_band_filters, int amount_spatial_filters_per_band, int amount_selected_features, int amount_labels){

	this->amount_electrodes = amount_electrodes;
	this->amount_time = amount_time;
	this->amount_band_filters = amount_band_filters;
	this->amount_spatial_filters_per_band = amount_spatial_filters_per_band;
	this->amount_selected_features = amount_selected_features;
	this->amount_labels = amount_labels;
	
	this->spatial_filters = new T**[this->amount_band_filters];
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		this->spatial_filters[bf] = new T*[this->amount_spatial_filters_per_band];
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			this->spatial_filters[bf][sf] = new T[this->amount_electrodes];
			for(int e = 0; e < this->amount_electrodes; e++){
				this->spatial_filters[bf][sf][e] = ((int)(e == sf));
			}
		}
	}
	
	this->mibif = new MIBIF<T>(this->amount_band_filters*this->amount_spatial_filters_per_band, this->amount_labels);
	this->classifier = new Naive_Bayes_Parzen_Window<T>(this->amount_selected_features, this->amount_labels);
}

template<typename T>
BCI<T>::~BCI(){

	for(int bf = 0; bf < this->amount_band_filters; bf++){
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			delete[] this->spatial_filters[bf][sf];
		}	delete[] this->spatial_filters[bf];
	}		delete[] this->spatial_filters;

	delete this->mibif;
	delete this->classifier;
	
}

template<typename T>
void BCI<T>::preprocessing(T** X, T*** X_bank, T band_pass_limits[][2], T acquisition_frequency, int full_time, int start_time){
// X[electrode][time], X_bank[id_filter][electrode][time], full_time = length of vector with signal, start_time = point the start the exit signal

	T* temp_wave = new T[full_time];
	T dt = 1.0/acquisition_frequency;

	for(int f = 0; f < this->amount_band_filters; f++){
		for(int e = 0; e < this->amount_electrodes; e++){
			for(int t = 0; t < full_time; t++) temp_wave[t] = X[e][t];
			band_pass_filter(band_pass_limits[f][0], band_pass_limits[f][1], temp_wave, full_time, dt);
			for(int t = 0; t < full_time-start_time; t++) X_bank[f][e][t] = temp_wave[t+start_time];
		}
	}
	
	delete[] temp_wave;
}

template<typename T>
void BCI<T>::spatial_filter(T*** X, T*** X_filtered){
// X[band][electrode][time], X_filtered[band][spatial_filter][time]
	
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			for(int t = 0; t < this->amount_time; t++){
				X_filtered[bf][sf][t] = X[bf][0][t]*this->spatial_filters[bf][sf][0];
				for(int e = 1; e < this->amount_electrodes; e++){
					X_filtered[bf][sf][t] += X[bf][e][t]*this->spatial_filters[bf][sf][e];
				}
			}
		}
	}
	
}

template<typename T>
void BCI<T>::log_power(T*** X, T* C){
// X_filtered[band][spatial_filter][time], C[amount_band*amount_spatial_filter]

	int start_position = 0;
	T traco;
	
	T **H = new T*[this->amount_spatial_filters_per_band];
	for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
		H[i] = new T[this->amount_spatial_filters_per_band];
	}	
	
	// log( diag(Z*Z^T)/tr(Z*Z^T) )
	for(int b = 0; b < this->amount_band_filters; b++){
		traco = 0.0;
		for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
			for(int j = 0; j < this->amount_spatial_filters_per_band; j++){
				H[i][j] = 0.0;
				for(int k = 0; k < this->amount_time; k++){
					H[i][j] += X[b][i][k]*X[b][j][k];
				}
				if(i == j) traco += H[i][j];
			}
		}
		for(int i = 0; i < this->amount_spatial_filters_per_band; i++) C[start_position+i] = log(H[i][i]/traco);
		start_position += this->amount_spatial_filters_per_band;
	}
	
	for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
		delete[] H[i];
	}	delete[] H;	
}

template<typename T>
void BCI<T>::paired_mibif(){

	int addition = 0;
	for(int i = 0; i < this->amount_selected_features+addition; i = i+2){
		if(this->mibif->get_id_variable(i)%2==0){
			if(this->mibif->get_new_position(this->mibif->get_id_variable(i)+1) >= this->amount_selected_features+addition) addition++;
			mibif->set_forced_position(this->mibif->get_id_variable(i)+1, i+1);
		}else{
			if(this->mibif->get_new_position(this->mibif->get_id_variable(i)-1) >= this->amount_selected_features+addition) addition++;
			mibif->set_forced_position(this->mibif->get_id_variable(i)-1, i+1);			
		}
	}
	if(addition != 0){
		delete this->classifier;
		this->classifier = new Naive_Bayes_Parzen_Window<T>(this->amount_selected_features+addition, this->amount_labels);
	}
	
}

template<typename T>
void BCI<T>::fit_preprocess_without_spatial(T***** X_bank, int amount_trial_per_label){
// X[label][id_trial][band][electrode][time]
	
	T*** X_transformed = new T**[this->amount_band_filters];
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		X_transformed[bf] = new T*[this->amount_spatial_filters_per_band];
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			X_transformed[bf][sf] = new T[this->amount_time];
		}
	}
	
	T** X_train = new T*[amount_trial_per_label*this->amount_labels];
	int* Y_train = new int[amount_trial_per_label*this->amount_labels];
	for(int i = 0; i < amount_trial_per_label*this->amount_labels; i++){
		X_train[i] = new T[this->amount_band_filters*this->amount_spatial_filters_per_band];
	}
	
	for(int l = 0; l < this->amount_labels; l++){
		for(int i = 0; i < amount_trial_per_label; i++){
			this->spatial_filter(X_bank[l][i], X_transformed);
			this->log_power(X_transformed, X_train[l*amount_trial_per_label+i]);
			Y_train[l*amount_trial_per_label+i] = l+1;
		}
	}
	
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			delete[] X_transformed[bf][sf];
		}	delete[] X_transformed[bf];
	}		delete[] X_transformed;
	
	this->mibif->fit(X_train, Y_train, amount_trial_per_label*this->amount_labels);
	#ifdef CSP_LIB
		this->paired_mibif();
	#endif
	
	this->mibif->transform(X_train, amount_trial_per_label*this->amount_labels);

	this->classifier->fit(X_train, Y_train, amount_trial_per_label*this->amount_labels);
	
	for(int i = 0; i < amount_trial_per_label*this->amount_labels; i++){
		delete[] X_train[i];
	}	delete[] X_train;
		delete[] Y_train;
	
}

template<typename T>
int BCI<T>::predict(T*** X_bank){
	
	T*** X_transformed = new T**[this->amount_band_filters];
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		X_transformed[bf] = new T*[this->amount_spatial_filters_per_band];
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			X_transformed[bf][sf] = new T[this->amount_time];
		}
	}
	T* X_train = new T[this->amount_band_filters*this->amount_spatial_filters_per_band];
	
	this->spatial_filter(X_bank, X_transformed);
	this->log_power(X_transformed, X_train);
	
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			delete[] X_transformed[bf][sf];
		}	delete[] X_transformed[bf];
	}		delete[] X_transformed;
	
	this->mibif->transform(X_train);
	int exit = this->classifier->predict(X_train);
	
	delete[] X_train;
	
	return exit;
}

template<typename T>
void BCI<T>::shift_spatial_filter(T*** X, T*** X_filtered, int shift){
// X[band][electrode][time], X_filtered[band][spatial_filter][time]
	
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			for(int t = 0; t < this->amount_time; t++){
				X_filtered[bf][sf][t] = X[bf][0][t+shift]*this->spatial_filters[bf][sf][0];
				for(int e = 1; e < this->amount_electrodes; e++){
					X_filtered[bf][sf][t] += X[bf][e][t+shift]*this->spatial_filters[bf][sf][e];
				}
			}
		}
	}
	
}

template<typename T>
int BCI<T>::shift_predict(T*** X_bank, int shift){
	
	T*** X_transformed = new T**[this->amount_band_filters];
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		X_transformed[bf] = new T*[this->amount_spatial_filters_per_band];
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			X_transformed[bf][sf] = new T[this->amount_time];
		}
	}
	T* X_train = new T[this->amount_band_filters*this->amount_spatial_filters_per_band];
	
	this->shift_spatial_filter(X_bank, X_transformed, shift);
	this->log_power(X_transformed, X_train);
	
	for(int bf = 0; bf < this->amount_band_filters; bf++){
		for(int sf = 0; sf < this->amount_spatial_filters_per_band; sf++){
			delete[] X_transformed[bf][sf];
		}	delete[] X_transformed[bf];
	}		delete[] X_transformed;
	
	this->mibif->transform(X_train);
	int exit = this->classifier->predict(X_train);
	
	delete[] X_train;
	
	return exit;
}

template<typename T>
T BCI<T>::zero_hold(T***** X_bank, int amount_trials_per_label, int max_position){
	
	int temp;
	T kappa = -1e+6;
	T temp_kappa;
	int **confusao = new int*[this->amount_labels];
	for(int i = 0; i < this->amount_labels; i++){
		confusao[i] = new int[this->amount_labels];
	}
	int **best_confusao = new int*[this->amount_labels];
	for(int i = 0; i < this->amount_labels; i++){
		best_confusao[i] = new int[this->amount_labels];
	}
	
	for(int t = 0; t+this->amount_time < max_position; t = t+10){ // avaliation for each 10 points interval 
		for(int i = 0; i < this->amount_labels; i++) for(int j = 0; j < this->amount_labels; j++) confusao[i][j] = 0;
		
		for(int l = 0; l < this->amount_labels; l++){
			for(int i = 0; i < amount_trials_per_label; i++){
				temp = this->shift_predict(X_bank[l][i], t);
				confusao[l][temp-1]++;
			}
		}
		temp_kappa = coeficiente_kappa(confusao, this->amount_labels);
		if(temp_kappa > kappa){
			for(int i = 0; i < this->amount_labels; i++) for(int j = 0; j < this->amount_labels; j++) best_confusao[i][j] = confusao[i][j];
			kappa = temp_kappa;
		}
	}
	for(int i = 0; i < this->amount_labels; i++){
		for(int j = 0; j < this->amount_labels; j++){
			printf("%2d  ", best_confusao[i][j]);
		}	printf("\n");
	}
	printf("KAPPA: %f\n", kappa);
	for(int i = 0; i < this->amount_labels; i++){
		delete[] confusao[i];
		delete[] best_confusao[i];
	}	delete[] best_confusao;
		delete[] confusao;
	
	return kappa;
	
}

template<typename T>
T BCI<T>::kappa_calculate(T***** X_bank, int amount_trials_per_label, int position){

	int temp;
	int acertos = 0;
	T kappa;
	int **confusao = new int*[this->amount_labels];
	for(int i = 0; i < this->amount_labels; i++){
		confusao[i] = new int[this->amount_labels];
		for(int j = 0; j < this->amount_labels; j++) confusao[i][j] = 0;
	}
	
	for(int l = 0; l < this->amount_labels; l++){
		for(int i = 0; i < amount_trials_per_label; i++){
			temp = this->shift_predict(X_bank[l][i], position);
			confusao[l][temp-1]++;
			if(temp == l+1) acertos++;
		}
	}
	printf("\nConfusao:\n");
	for(int i = 0; i < this->amount_labels; i++){
		for(int j = 0; j < this->amount_labels; j++) printf("\t%2d", confusao[i][j]);
		printf("\n");
	}
	kappa = coeficiente_kappa(confusao, this->amount_labels);
	printf("\nKappa: %.6lf\n", kappa);
	printf("Acuracia: %.6lf\n\n", acertos/(1.0*this->amount_labels*amount_trials_per_label));
	
	
	for(int i = 0; i < this->amount_labels; i++){
		delete[] confusao[i];
	}	delete[] confusao;
	
	return kappa;	
}

template<typename T>
T BCI<T>::crossentropy_calculate(T***** X_bank, int amount_trials_per_label, int position){

	T crossentropy = 0;

	for(int l = 0; l < this->amount_labels; l++){
		for(int i = 0; i < amount_trials_per_label; i++){
			this->shift_predict(X_bank[l][i], position);
			for(int c = 0; c < this->amount_labels; c++){
				if(c==l) crossentropy -= log(this->classifier->get_fitness(c+1));
				else crossentropy -= log(1-this->classifier->get_fitness(c+1));
			}
		}
	}

	return crossentropy/(this->amount_labels*amount_trials_per_label);

}

#endif // BCI_H


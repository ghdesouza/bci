#ifndef FBCSP_H
#define FBCSP_H

#include "naive_bayes_parzen_window.tpp"
#include "fourier_transform.tpp"
#include "csp.tpp"
#include "mibif.tpp"

class FBCSP : public Classifiers<float>{

    protected:
        int amount_electrodes;
        int amount_time;
        float frequency;
        // int amount_labels; // included in father
        
        int amount_temporal_filters;
        float** temporal_filter;
        int amount_spatial_filters_per_band;
        float*** spatial_filter; // [amount_temporal_filters][amount_spatial_filters_per_band][amount_electrodes]

        int amount_features;
        
        MIBIF<float>* mibif;
        Classifiers<float>* classifier;

        void to_matriz(float* s, float** S);
        void temporal_transform(int id_temporal_filter, float** S);
        void spatial_transform(int id_temporal_filter, float** S, float**Z);
        void feature_extractor(float** Z, float* X);  // LogPower Function
        
    public:
        FBCSP(string file_name);
        ~FBCSP();

        int get_amount_electrodes(){return this->amount_electrodes;}
        int get_amount_time(){return this->amount_time;}
        int get_amount_temporal_filters(){return this->amount_temporal_filters;}
        int get_amount_spatial_filters(){return this->amount_spatial_filters_per_band;}
        int get_amount_features(){return this->amount_features;}
        
        void fit(float** data, int* labels, int data_size);
        void get_features(float* s, float* features);
        
        int predict(float* s);
        float get_fitness(int label){return this->classifier->get_fitness(label);}
        
        int predict_from_feature(float* features){return this->classifier->predict(features);}
        
        void get_features_multiwindow(float* s, float** features, int time_size);
        int predict_multiwindow(float* s, int time_size);
        metrics full_metrics_multiwindows(float** data, int* labels, int data_size, int time_size);
};

FBCSP::FBCSP(string file_name){
    
	FILE* arq = fopen(file_name.c_str(), "r");
    if(!arq){ printf("ERROR (FBCSP): File not found!\n"); return; }
    
    char *trash = new char[40];
    
    // dataset informations
    fscanf(arq, "%s", trash); fscanf(arq, "%d", &this->amount_electrodes);
    fscanf(arq, "%s", trash); fscanf(arq, "%d", &this->amount_time);
    fscanf(arq, "%s", trash); fscanf(arq, "%f", &this->frequency);
    fscanf(arq, "%s", trash); fscanf(arq, "%d", &this->amount_labels);
        
    // classifier informations
    fscanf(arq, "%s", trash); fscanf(arq, "%d", &this->amount_temporal_filters);
    fscanf(arq, "%s", trash); fscanf(arq, "%d", &this->amount_spatial_filters_per_band); // 2*m
    fscanf(arq, "%s", trash); fscanf(arq, "%d", &this->amount_features);

    // temporal filter intevals
    fscanf(arq, "%s", trash);
    this->temporal_filter = new float*[this->amount_temporal_filters];
    for(int i = 0; i < this->amount_temporal_filters; i++){
        this->temporal_filter[i] = new float[2];
        fscanf(arq, "%f", &this->temporal_filter[i][0]);
        fscanf(arq, "%f", &this->temporal_filter[i][1]);
    }
    
    this->spatial_filter = new float**[this->amount_temporal_filters];
    for(int i = 0; i < this->amount_temporal_filters; i++){
        this->spatial_filter[i] = new float*[this->amount_spatial_filters_per_band];
        for(int j = 0; j < this->amount_spatial_filters_per_band; j++){
            this->spatial_filter[i][j] = new float[this->amount_electrodes];
        }
    }
 
    delete[] trash;
 
    this->mibif = new MIBIF<float>(this->amount_temporal_filters*this->amount_spatial_filters_per_band, this->amount_labels);
    this->classifier = new Naive_Bayes_Parzen_Window<float>(this->amount_features, this->amount_labels);
    
}

FBCSP::~FBCSP(){
    
    for(int i = 0; i < this->amount_temporal_filters; i++){
        delete[] this->temporal_filter[i];
    }   delete[] this->temporal_filter;
    
    for(int i = 0; i < this->amount_temporal_filters; i++){
        for(int j = 0; j < this->amount_spatial_filters_per_band; j++){
            delete[] this->spatial_filter[i][j];
        }   delete[] this->spatial_filter[i];
    }       delete[] this->spatial_filter;

    delete this->mibif;
    delete this->classifier;
}

void FBCSP::to_matriz(float* s, float** S){
    
    int temp_int = 0;
    for(int j = 0; j < this->amount_time; j++){
        for(int i = 0; i < this->amount_electrodes; i++){
            S[i][j] = s[temp_int];
            temp_int++;
        }
    }
}

void FBCSP::temporal_transform(int id_temporal_filter, float** S){
        for(int i = 0; i < this->amount_electrodes; i++){
            band_pass_filter(this->temporal_filter[id_temporal_filter][0], this->temporal_filter[id_temporal_filter][1], S[i], this->amount_time, 1/this->frequency);
        }
}

void FBCSP::spatial_transform(int id_temporal_filter, float** S, float**Z){
    for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
        for(int j = 0; j < this->amount_time; j++){
            Z[i][j] = 0;
            for(int l = 0; l < this->amount_electrodes; l++){
                   Z[i][j] += S[l][j]*this->spatial_filter[id_temporal_filter][i][l];
            }
        }
    }
}

void FBCSP::feature_extractor(float** Z, float* X){
// Z[amount_spatial_filters_per_band][amount_time], X[amount_spatial_filters_per_band]
	
    float traco = 0.0;
    float* H = new float[this->amount_spatial_filters_per_band];
	
    //X = log( diag(Z*Z^T)/tr(Z*Z^T) )
    for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
        H[i] = 0.0;
        for(int k = 0; k < this->amount_time; k++){
            H[i] += Z[i][k]*Z[i][k];
        }
        traco += H[i];
	}
	for(int i = 0; i < this->amount_spatial_filters_per_band; i++) X[i] = log(H[i]/traco);
	delete[] H;    
    
}

void FBCSP::fit(float** data, int* labels, int data_size){
    
    float*** data_csp;
    float** data_features;
    float** Z;
    float* X_temp;
    
    { // alloc
        data_features = new float*[data_size];
        data_csp = new float**[data_size];

        for(int i = 0; i < data_size; i++){
            data_features[i] = new float[this->amount_temporal_filters*this->amount_spatial_filters_per_band];
            
            data_csp[i] = new float*[this->amount_electrodes];
            for(int j = 0; j < this->amount_electrodes; j++){
                data_csp[i][j] = new float[this->amount_time];
            }
        }
        
        Z = new float*[this->amount_spatial_filters_per_band];
        for(int i = 0; i < this->amount_spatial_filters_per_band; i++) Z[i] = new float[this->amount_time];
        X_temp = new float[this->amount_spatial_filters_per_band];
        
    }
    
    for(int i = 0; i < this->amount_temporal_filters; i++){
        
        for(int j = 0; j < data_size; j++){
            this->to_matriz(data[j], data_csp[j]);
            this->temporal_transform(i, data_csp[j]);
        }
        
        csp_spatial_filter(data_csp, labels, data_size, this->amount_spatial_filters_per_band, this->amount_electrodes, this->amount_time, this->spatial_filter[i]);
        
        for(int j = 0; j < data_size; j++){
            this->spatial_transform(i, data_csp[j], Z);
            this->feature_extractor(Z, X_temp);
            
            for(int k = 0; k < this->amount_spatial_filters_per_band; k++){
                data_features[j][this->amount_spatial_filters_per_band*i+k] = X_temp[k];
            }
        }
    }
    
    this->mibif->fit(data_features, labels, data_size);
    this->mibif->transform(data_features, data_size);
    
    this->classifier->fit(data_features, labels, data_size);    
    
    { // desalloc
        
        for(int i = 0; i < data_size; i++){
            delete[] data_features[i];
        }   delete[] data_features;
        for(int i = 0; i < data_size; i++){
            for(int j = 0; j < this->amount_electrodes; j++){
                delete[] data_csp[i][j];
            }   delete[] data_csp[i];
        }       delete[] data_csp;
        for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
            delete[] Z[i];
        }   delete[] Z;
        delete[] X_temp;
    }
    
}

void FBCSP::get_features(float* s, float* features){
    
    float** S; float** Z; float* X; float* X_temp; int y;
    
    { // alloc
        S = new float*[this->amount_electrodes];
        for(int i = 0; i < this->amount_electrodes; i++) S[i] = new float[this->amount_time];
        Z = new float*[this->amount_spatial_filters_per_band];
        for(int i = 0; i < this->amount_spatial_filters_per_band; i++) Z[i] = new float[this->amount_time];
        X_temp = new float[this->amount_spatial_filters_per_band];
        X = new float[this->amount_temporal_filters*this->amount_spatial_filters_per_band];
    }
    
    for(int i = 0; i < this->amount_temporal_filters; i++){
        this->to_matriz(s, S);
        this->temporal_transform(i, S);
        this->spatial_transform(i, S, Z);
        this->feature_extractor(Z, X_temp);
        for(int j = 0; j < this->amount_spatial_filters_per_band; j++){
            X[this->amount_spatial_filters_per_band*i+j] = X_temp[j];
        }
    }
    this->mibif->transform(X);
    
    for(int i = 0; i < this->amount_temporal_filters*this->amount_spatial_filters_per_band; i++){
        features[i] = X[i];
    }
    
    { // desalloc
        for(int i = 0; i < this->amount_electrodes; i++){
            delete[] S[i];
        }   delete[] S;
        for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
            delete[] Z[i];
        }   delete[] Z;
        delete[] X_temp;
        delete[] X;
    }
    
}

int FBCSP::predict(float* s){
    
    float* X; int y;
    X = new float[this->amount_temporal_filters*this->amount_spatial_filters_per_band];
    
    this->get_features(s, X);
    y = this->classifier->predict(X);
    delete[] X;
    
    return y;
    
}

void FBCSP::get_features_multiwindow(float* s, float** features, int time_size){
    
    float** S; float** Z; float** X; float* X_temp; float** Z_window;
    int temp_int;
    
    { // alloc
        S = new float*[this->amount_electrodes];
        for(int i = 0; i < this->amount_electrodes; i++) S[i] = new float[time_size];
        Z = new float*[this->amount_spatial_filters_per_band];
        for(int i = 0; i < this->amount_spatial_filters_per_band; i++) Z[i] = new float[time_size];
        Z_window = new float*[this->amount_spatial_filters_per_band];
        for(int i = 0; i < this->amount_spatial_filters_per_band; i++) Z_window[i] = new float[this->amount_time];
        X_temp = new float[this->amount_spatial_filters_per_band];
        X = new float*[ ((int)(4*time_size/this->frequency))];
        for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++) X[j] = new float[this->amount_temporal_filters*this->amount_spatial_filters_per_band];
    }
    
    for(int i = 0; i < this->amount_temporal_filters; i++){
    
        // to_matrix
        temp_int = 0;
        for(int k = 0; k < time_size; k++){
            for(int j = 0; j < this->amount_electrodes; j++){
                S[j][k] = s[temp_int];
                temp_int++;
            }
        }
        
        // temporal_transform
        for(int j = 0; j < this->amount_electrodes; j++){
            band_pass_filter(this->temporal_filter[i][0], this->temporal_filter[i][1], S[j], time_size, 1/this->frequency);
        }
        
        // spatial_transform
        for(int j = 0; j < this->amount_spatial_filters_per_band; j++){
            for(int k = 0; k < time_size; k++){
                Z[j][k] = 0;
                for(int l = 0; l < this->amount_electrodes; l++){
                    Z[j][k] += S[l][k]*this->spatial_filter[i][j][l];
                }
            }
        }
        
        // feature_extraction
        for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++){
            for(int a = 0; a < this->amount_spatial_filters_per_band; a++){
                for(int b = 0; b < this->amount_time; b++){
                    Z_window[a][b] = Z[a][b+j*((int) this->frequency/4)];
                }
            }
            this->feature_extractor(Z_window, X_temp);
            for(int k = 0; k < this->amount_spatial_filters_per_band; k++){
                X[j][this->amount_spatial_filters_per_band*i+k] = X_temp[k];
            }
        }
        
    }
    
    for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++){
        this->mibif->transform(X[j]);
    }
    
    for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++){
        for(int i = 0; i < this->amount_temporal_filters*this->amount_spatial_filters_per_band; i++){
            features[j][i] = X[j][i];
        }
    }
    
    { // desalloc
        for(int i = 0; i < this->amount_electrodes; i++){
            delete[] S[i];
        }   delete[] S;
        for(int i = 0; i < this->amount_spatial_filters_per_band; i++){
            delete[] Z[i];
            delete[] Z_window[i];
        }   delete[] Z;
            delete[] Z_window;
        delete[] X_temp;
        for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++){
            delete[] X[j];
        }   delete[] X;
    }
    
}

int FBCSP::predict_multiwindow(float* s, int time_size){
    
    int y;
    float* probs = new float[this->amount_labels];
    for(int i = 0; i < this->amount_labels; i++) probs[i] = 0;
    
    float** X = new float*[((int) (((time_size-this->amount_time)/(this->frequency/4))+1))];
    for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++) X[j] = new float[this->amount_temporal_filters*this->amount_spatial_filters_per_band];
    
    this->get_features_multiwindow(s, X, time_size);
    
    for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++){
        y = this->classifier->predict(X[j]);
        for(int i = 0; i < this->amount_labels; i++){
            #ifndef MAX
                if(probs[i] < get_fitness(i+1)) probs[i] = get_fitness(i+1); //max
            #else
                probs[i] += get_fitness(i+1); //integrate
            #endif
        }
    }
    
	y = 0;
	float best_prob = probs[0];
    for(int i = 1; i < this->amount_labels; i++){
    	if(probs[i] > best_prob){
			y = i;
			best_prob = probs[i];
		}
	}
	
    for(int j = 0; j < ((int) (((time_size-this->amount_time)/(this->frequency/4))+1)); j++){
        delete[] X[j];
    }   delete[] X;
        delete[] probs;
    
    return (y+1);
}

metrics FBCSP::full_metrics_multiwindows(float** data, int* labels, int data_size, int time_size){
    
	float kappa, mean_squared_error, crossentropy, accurace, auc;
	kappa = 0; mean_squared_error = 0; crossentropy = 0; accurace = 0; auc = 0;
	int temp_class;
	float temp_val;
    
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
	printf("\n");
	for(int i = 0; i < data_size; i++){
		temp_class = this->predict_multiwindow(data[i], time_size);
        printf("%d, ", temp_class);
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

#endif // FBCSP_H


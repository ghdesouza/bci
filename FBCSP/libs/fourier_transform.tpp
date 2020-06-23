#ifndef FOURIER_TRANSFORM_H
#define FOURIER_TRANSFORM_H

template<typename T>
void fourier_transform(T* time_domain, T** frequency_domain, int amount_step, T dt){

    //amount of frequency to calculate
    int amount_frequency = (int)(amount_step/2);

    T b_k;

    //calculate the frequencies
    for(int f = 0; f < amount_frequency; f++){
        frequency_domain[f][0] = f/((amount_step)*dt);
        frequency_domain[f][1] = frequency_domain[f][2] = 0;
    }

    //calculating the values of the discrete fourier transform
    T temp_bk;
    for(int cf = 0; cf < amount_frequency; cf++){
        temp_bk = (M_PI * 2.0 * cf);
        for(int ct = 0; ct < amount_step; ct++){
            b_k = (temp_bk * ct)/amount_step;
            frequency_domain[cf][1] += 2*time_domain[ct]*cos(-b_k);
            frequency_domain[cf][2] += 2*time_domain[ct]*sin(-b_k);
        }
	}
}

template<typename T>
void inverse_fourier_transform(T** frequency_domain, int amount_step, T* time_domain){

    T b_k;

    T temp_bk;
    for(int ct = 0; ct < amount_step; ct++){
        time_domain[ct] = 0;
        temp_bk = (M_PI * 2.0 * ct);
        for(int cf = 0; cf < (int)(amount_step/2); cf++){
            b_k = (temp_bk * cf)/amount_step;
            time_domain[ct] += (frequency_domain[cf][1]*cos(b_k)-frequency_domain[cf][2]*sin(b_k));
            }
        time_domain[ct] /= amount_step;
    }
}

template<typename T>
void band_pass_filter(T low_frequency, T high_frequency, T *wave, int amount_step, T dt){

    int amount_frequency = (int)(amount_step/2);
	T **frequency_domain = new T*[amount_frequency];
	for(int i = 0; i < amount_frequency; i++) frequency_domain[i] = new T[3];

    fourier_transform(wave, frequency_domain, amount_step, dt);

    int i = 0;
    for(;frequency_domain[i][0] < low_frequency && i < amount_frequency; i++) frequency_domain[i][1] = frequency_domain[i][2] = 0;
    for(;frequency_domain[i][0] <= high_frequency && i < amount_frequency; i++);
    for(;i < amount_frequency; i++) frequency_domain[i][1] = frequency_domain[i][2] = 0;
    
    inverse_fourier_transform(frequency_domain, amount_step, wave);
    
    for(int i = 0; i < amount_frequency; i++){
		delete[] frequency_domain[i];
	}
    delete[] frequency_domain;
}

#endif // FOURIER_TRANSFORM_H


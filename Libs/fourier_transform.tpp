/**
 * @file fourier_transform.tpp
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

#ifndef FOURIER_TRANSFORM_H
#define FOURIER_TRANSFORM_H

#include <math.h>

template<typename T>
void fourier_transform(T* time_domain, T** frequency_domain, int amount_step, T dt){

    //amount of frequency to calculate
    int amount_frequency = (int)(amount_step/2);

    T b_k;

    //calculate the frequencies
    for(int f = 0; f < amount_frequency; f++){
        frequency_domain[f][0] = f/((amount_step)*dt);
        frequency_domain[f][1] = 0;
        frequency_domain[f][2] = 0;
    }

    //calculating the values of the discrete fourier transform
    for(int cf = 0; cf < amount_frequency; cf++){
        for(int ct = 0; ct < amount_step; ct++){
            b_k = (M_PI * 2.0 * cf * ct)/amount_step;
            frequency_domain[cf][1] += 2*time_domain[ct]*cos(-b_k);
            frequency_domain[cf][2] += 2*time_domain[ct]*sin(-b_k);
        }
	}
}

template<typename T>
void inverse_fourier_transform(T** frequency_domain, int amount_step, T* time_domain){

    T b_k;

    for(int ct = 0; ct < amount_step; ct++){
        time_domain[ct] = 0;
        for(int cf = 0; cf < (int)(amount_step/2); cf++){
            b_k = (M_PI*2.0*cf*ct)/amount_step;
            time_domain[ct] += (frequency_domain[cf][1]*cos(b_k)-frequency_domain[cf][2]*sin(b_k));
            }
        time_domain[ct] /= amount_step;
    }
}

template<typename T>
void band_pass_filter(T low_frequency, T high_frequency, T *wave, int amount_step, T dt){

	T **frequency_domain = new T*[(int)(amount_step/2)];
	for(int i = 0; i < (int)(amount_step/2); i++) frequency_domain[i] = new T[3];

    fourier_transform(wave, frequency_domain, amount_step, dt);

    for(int i = 0; i < (int)(amount_step/2); i++){
        if(frequency_domain[i][0] > high_frequency || frequency_domain[i][0] < low_frequency){
            frequency_domain[i][1] = 0;
            frequency_domain[i][2] = 0;
        }
	}

    inverse_fourier_transform(frequency_domain, amount_step, wave);
    
    for(int i = 0; i < (int)(amount_step/2); i++){
		delete[] frequency_domain[i];
	}
    delete[] frequency_domain;
}

#endif // FOURIER_TRANSFORM_H


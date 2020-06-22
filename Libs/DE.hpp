/**
 * @file DE.hpp
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

#ifndef EVOLUCAO_DIFERENCIAL_H
#define EVOLUCAO_DIFERENCIAL_H

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>

using namespace std;

class Evolucao_Diferencial{

    private:

        int dimensao;
        float convergencia_objetivo;
        float convergencia_populacao;
        float escala_vetor;
        float prob_mudar;
        int tam_populacao;
        bool tem_restricao;
        bool tem_busca_local;
        float **populacao;
        float (*criterio)(float*, int);
        void (*verifica_corrige_restricao)(float*, int);
        void (*busca_Local)(float*, int);
        float prob_bl;

        void cruzamento();

    public:

        Evolucao_Diferencial(int dimen, int tam_pop, float (*crit)(float*, int), float escal_vet, float prob_mud);
        ~Evolucao_Diferencial();

        void proxima_geracao();
        void set_criterio(float (*crit)(float*, int));
        void mostra_populacao();
        void mostra_criterios();
        float* get_melhor_individuo();
        float get_convergencia_objetivo(){return convergencia_objetivo;}
        float get_convergencia_populacao(){return convergencia_populacao;}
        void set_busca_local(void (*bp)(float*, int), float percent);
        void set_restricao(void (*restringe_corrige)(float*, int));
        void salva_criterios(string name_file);
        void salva_melhor_individuo(string name_file);
        void salva_populacao(string name_file);
        void carrega_populacao(string name_file);
};

Evolucao_Diferencial::Evolucao_Diferencial(int dimen, int tam_pop, float (*crit)(float*, int), float escal_vet, float prob_mud){
    escala_vetor = escal_vet;
    dimensao = dimen;
    tam_populacao = tam_pop;
    prob_mudar = prob_mud;
    convergencia_populacao = -1.0;
    convergencia_objetivo = -1.0;
    tem_busca_local = false;
    tem_restricao = false;

    criterio = crit;

    populacao = new float*[(2*tam_populacao)];
    for(int i = 0; i < (2*tam_populacao); ++i){
        populacao[i] = new float[dimensao+1]; // ultima posicao possui o valor criterio
        for(int j = 0; j < dimensao; j++) populacao[i][j] = 2.0*((float) rand()/RAND_MAX) - 1.0;
        populacao[i][dimensao] = criterio(populacao[i], dimensao);
        if(tem_restricao){
            verifica_corrige_restricao(populacao[i], dimensao);
            populacao[i][dimensao] = criterio(populacao[i], dimensao);

        }
    }

}

Evolucao_Diferencial::~Evolucao_Diferencial(){

    for (int i = 0; i < (2*tam_populacao); ++i){
        delete[] populacao[i];
    }
    delete[] populacao;

}

void Evolucao_Diferencial::proxima_geracao(){
    cruzamento();
}

void Evolucao_Diferencial::set_criterio(float (*crit)(float*, int)){
    criterio = crit;
    for(int i = 0; i < tam_populacao; i++) populacao[i][dimensao] = criterio(populacao[i], dimensao);
}

void Evolucao_Diferencial::cruzamento(){
    int pai_1, pai_2, pai_3, imutavel;
    int melhor_filho;
    float temp;
    bool igual;
    float norma_1, diferenca;
    convergencia_populacao = 0.0;
    convergencia_objetivo = 0.0;

    for(int i = tam_populacao; i < 2*tam_populacao; i++){

        pai_1 = ((int)rand())%tam_populacao;
        pai_2 = ((int)rand())%tam_populacao;
        pai_3 = ((int)rand())%tam_populacao;
        imutavel = ((int)rand())%(dimensao);
        igual = (pai_1 == pai_2);

        norma_1 = 0.0;
        for(int j = 0; j < dimensao; j++){
            diferenca = populacao[pai_1][j] - populacao[pai_2][j];
            norma_1 += fabs(diferenca);
            if(((((float) rand()/RAND_MAX) < prob_mudar) || (j == imutavel)) && (!igual)){
                populacao[i][j] = populacao[pai_3][j] + escala_vetor*(diferenca);
            }else populacao[i][j] = populacao[i-tam_populacao][j];
        }
        if(tem_restricao) verifica_corrige_restricao(populacao[i], dimensao);
        populacao[i][dimensao] = criterio(populacao[i], dimensao);
        convergencia_populacao += (norma_1/dimensao);

    }
    convergencia_populacao /= tam_populacao;

    if(tem_busca_local){
        melhor_filho = tam_populacao;
        temp = populacao[melhor_filho][dimensao];
        for(int i = tam_populacao+1; i < 2*tam_populacao; i++) if(populacao[i][dimensao] < melhor_filho){
            melhor_filho = i;
            temp = populacao[melhor_filho][dimensao];
        }
        busca_Local(populacao[melhor_filho], dimensao);
        populacao[melhor_filho][dimensao] = criterio(populacao[melhor_filho], dimensao);
    }

    for(int i = 0; i < tam_populacao; i++){
        if(populacao[tam_populacao+i][dimensao] <= populacao[i][dimensao]){
            swap(populacao[tam_populacao+i], populacao[i]);
        }
        convergencia_objetivo += populacao[i][dimensao];
    }
    convergencia_objetivo /= tam_populacao;
}

void Evolucao_Diferencial::set_busca_local(void (*bp)(float*, int), float prob){
    tem_busca_local = true;
    busca_Local = bp;
    prob_bl = prob;
}

void Evolucao_Diferencial::set_restricao(void (*restringe_corrige)(float*, int)){
    tem_restricao = true;
    verifica_corrige_restricao = restringe_corrige;

    for(int i = 0; i < 2*tam_populacao; i++){
        verifica_corrige_restricao(populacao[i], dimensao);
        populacao[i][dimensao] = criterio(populacao[i], dimensao);
    }
}

void Evolucao_Diferencial::mostra_populacao(){
    cout << endl;
    for(int i = 0; i < tam_populacao; i++){
    cout << "Criterio: " << populacao[i][dimensao];
        for(int j = 0; j < dimensao; j++){
            cout << "\t" << populacao[i][j];
        }
    cout << endl;
    }
    cout << endl;
}

void Evolucao_Diferencial::mostra_criterios(){
    cout << endl;
    for(int i = 0; i < tam_populacao; i++){
    cout << "Criterio individuo " << i << ": " << populacao[i][dimensao] << endl;
    }
    cout << endl;
}

float* Evolucao_Diferencial::get_melhor_individuo(){
    int mel = 0;
    for(int i = 1; i < tam_populacao; i++) if(populacao[i][dimensao] <= populacao[mel][dimensao]) mel = i;
    return populacao[mel];
}

void Evolucao_Diferencial::salva_criterios(string name_file){
    ofstream arquivo;
    arquivo.open(name_file.c_str(), ios::app);

    for(int i = 0; i < tam_populacao; i++){
    arquivo << populacao[i][dimensao] << endl;
    }

    arquivo.close();
}

void Evolucao_Diferencial::salva_melhor_individuo(string name_file){

    int mel = 0;
    for(int i = 1; i < tam_populacao; i++) if(populacao[i][dimensao] <= populacao[mel][dimensao]) mel = i;

    ofstream arquivo;
    arquivo.open(name_file.c_str(), ios::ate);
    for(int i = 0; i < dimensao; i++) arquivo << populacao[mel][i] << "\n";
    arquivo.close();
}

void Evolucao_Diferencial::salva_populacao(string name_file){

    ofstream arquivo;
    arquivo.open(name_file.c_str(), ios::ate);
    for(int i = 0; i < dimensao; i++){
        for(int j = 0; j < tam_populacao; j++){
            arquivo << populacao[j][i];
            (j == tam_populacao-1) ? arquivo << endl : arquivo << "\t";
        }

    }
    arquivo.close();
}

void Evolucao_Diferencial::carrega_populacao(string name_file){

    ifstream arquivo;
    arquivo.open(name_file.c_str());
    for(int i = 0; i < dimensao; i++){
        for(int j = 0; j < tam_populacao; j++){
            arquivo >> populacao[j][i];
        }
    }
    for(int j = 0; j < tam_populacao; j++) populacao[j][dimensao] = criterio(populacao[j], dimensao);

    arquivo.close();
}

#endif // EVOLUCAO_DIFERENCIAL_H


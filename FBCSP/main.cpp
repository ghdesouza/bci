#include <fstream>

#define BCI FBCSP

#include "libs/dataset.tpp"
#include "libs/fbcsp.hpp"

using namespace std;

int main(int argc, char *argv[]){
    
    srand(2020);
    string config_name, eeg_train, eeg_test, exit_name;
    config_name = "fbcsp.conf"; eeg_train = argv[1]; eeg_test = argv[2]; exit_name = argv[3];

    BCI* bci;
    bci = new BCI(config_name);
    
    int data_size, trial_size;
    float **data;
    int *labels;
 
    FILE* file;
    metrics bci_metrics;
    
	{ // creating train dataset
        data_size = get_file_size(eeg_train); if(data_size == -1) return 0;
        trial_size = bci->get_amount_electrodes()*bci->get_amount_time();
        labels = new int[data_size];
        data = new float*[data_size];
        for(int i = 0; i < data_size; i++) data[i] = new float[trial_size];
		load_dataset(data, labels, trial_size, data_size, eeg_train.c_str()); // loading dataset
	}
    
    bci->fit(data, labels, data_size); // training
    
	{ // deleting dataset
		delete[] labels;
		for(int i = 0; i < data_size; i++) delete[] data[i]; 
		delete[] data;
	}
    { // creating test dataset
        data_size = get_file_size(eeg_test);
        trial_size = bci->get_amount_electrodes()*2304;
        labels = new int[data_size];
        data = new float*[data_size];
        for(int i = 0; i < data_size; i++) data[i] = new float[trial_size];
		load_dataset(data, labels, trial_size, data_size, eeg_test.c_str()); // loading dataset
	}
	
    file = fopen(exit_name.c_str(), "app");
    int pred;
    for(int i = 0; i < data_size; i++){
        pred = bci->predict_multiwindow(data[i], trial_size/bci->get_amount_electrodes());
        printf("%d  %d\n", pred, labels[i]);
        fprintf(file, "%d,%d\n", i+1, pred);
    }
    fclose(file);
    
	{ // deleting dataset
		delete[] labels;
		for(int i = 0; i < data_size; i++) delete[] data[i]; 
		delete[] data;
	}    
	
	delete bci;
	return 0;
}

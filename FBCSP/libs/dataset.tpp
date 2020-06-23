#ifndef DATASET_TPP
#define DATASET_TPP

#include <fstream>

using namespace std;

int get_file_size(string name_file){
    int number_of_lines = 0;
    int words = 0;
    string line;
    ifstream myfile(name_file.c_str());
    if(myfile.is_open()){
        getline(myfile, line);
        number_of_lines++;
        for(int i = 0; line[i] != '\0'; i++){
            if(line[i] == '\t' || line[i] == ' ') words++;
        } words++;
        while(!myfile.eof()){
            getline(myfile, line);
            number_of_lines++;
        }
        myfile.close();
    }
    else printf("ERROR (DATASET): File not found!\n");
    
    return number_of_lines-1;
}

void load_dataset(float** data, int* labels, int dimension, int data_size, string name_file){

	FILE* arq = fopen(name_file.c_str(), "r");
    if(!arq){
        printf("ERROR (DATASET): File not found!\n");
        return;
    }
	for(int i = 0; i < data_size; i++){
		for(int j = 0; j < dimension; j++){
			fscanf(arq, "%f", &data[i][j]);
		}
		fscanf(arq, "%d", &labels[i]);
	}
	fclose(arq);
	return;
}

void shuffle_data(float** data, int* labels, int data_size){
	
	int* order = new int[data_size];
	int pos_max, max_val, temp_int;
	float* temp_T;
		
	for(int i = 0; i < data_size; i++) order[i] = (int)rand();

	for(int i = 0; i < data_size; i++){
		pos_max = i;
		max_val = order[i];
		for(int j = i+1; j < data_size; j++){
			if(order[j] > max_val){
				pos_max = j;
				max_val = order[j];
			}
		}
		
		{ // swaps
			// trial
			temp_T = data[i];
			data[i] = data[pos_max];
			data[pos_max] = temp_T;
			// label
			temp_int = labels[i];
			labels[i] = labels[pos_max];
			labels[pos_max] = temp_int;
			// randon vector
			temp_int = order[i];
			order[i] = order[pos_max];
			order[pos_max] = temp_int;
		}
	}
	
	delete[] order;
}

void setfold_data(float** data, float** data_train, float** data_stop, float** data_test, 
				  int* labels, int* labels_train, int* labels_stop, int* labels_test, 
				  int data_size, int kfold_size, int id_fold){
	
	int block_size = (int) (data_size/kfold_size);
	int start_position = id_fold*block_size;
	
	// test block
	for(int i = 0; i < block_size; i++){
		data_test[i] = data[ (start_position+i)%data_size ];
		labels_test[i] = labels[ (start_position+i)%data_size ];		
	}	start_position += block_size;
	// stop block
	for(int i = 0; i < block_size; i++){
		data_stop[i] = data[ (start_position+i)%data_size ];
		labels_stop[i] = labels[ (start_position+i)%data_size ];		
	}	start_position += block_size;	
	// train block
	for(int i = 0; i < (data_size-2*block_size); i++){
		data_train[i] = data[ (start_position+i)%data_size ];
		labels_train[i] = labels[ (start_position+i)%data_size ];
	}
}

void setfold_data(float** data, float** data_train, float** data_test, 
				  int* labels, int* labels_train, int* labels_test, 
				  int data_size, int kfold_size, int id_fold){
	
	int block_size = (int) (data_size/kfold_size);
	int start_position = id_fold*block_size;
	
	// test block
	for(int i = 0; i < block_size; i++){
		data_test[i] = data[ (start_position+i)%data_size ];
		labels_test[i] = labels[ (start_position+i)%data_size ];		
	}	start_position += block_size;
	// train block
	for(int i = 0; i < (data_size-block_size); i++){
		data_train[i] = data[ (start_position+i)%data_size ];
		labels_train[i] = labels[ (start_position+i)%data_size ];
	}
}

#endif // DATASET_TPP



#define amount_full_time 1875
#define start_time 875
#define amount_classifier_time 500
#define acquisition_frequency 250.0

#define ZOH true
#define kfold_size 10
#define amount_kfold 5

int amount_test_per_label;
int amount_train_per_label;

FILE* file;
float**** data; //[label][trial][electrode][full_time]
float***** data_extended; //[label][trial][band][electrode][full_time-start_time]
float***** data_extended_train;
float***** data_extended_test;


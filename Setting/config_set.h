int amount_suject = 3;
int amount_labels = 4;
string names_datasets[4][3] = {{"Datas/subject_1_class_1.txt", "Datas/subject_2_class_1.txt", "Datas/subject_3_class_1.txt"},
                               {"Datas/subject_1_class_2.txt", "Datas/subject_2_class_2.txt", "Datas/subject_3_class_2.txt"},
							   {"Datas/subject_1_class_3.txt", "Datas/subject_2_class_3.txt", "Datas/subject_3_class_3.txt"},
							   {"Datas/subject_1_class_4.txt", "Datas/subject_2_class_4.txt", "Datas/subject_3_class_4.txt"}};
int amount_trials_per_label[3] = {80, 80, 80};

int amount_electrodes = 3;
int amount_spatial_filters_per_band = 2;
int amount_band_filters = 9;
float band_pass[9][2] = {{4.0, 8.0}, {8.0, 12.0}, {12.0, 16.0}, {16.0, 20.0}, {20.0, 24.0}, {24.0, 28.0}, {28.0, 32.0}, {32.0, 36.0}, {36.0, 40.0}};
int selected_features = 4;

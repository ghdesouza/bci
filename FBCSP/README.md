
before start this tutorial is needed to put the eigen lib in the lib folder.  
The eigen lib can be downloaded by http://eigen.tuxfamily.org/  

For run the FBCSP approach can be used the command:  

# g++ main.cpp -o main.out  
# main.out config_name eeg_train.txt eeg_test.txt exit_name.txt  

* eeg_train.txt is the name of the data file that will be used to fit the model  

* eeg_test.txt is the name of the data file that will be used to be predicted  

* exit_name.txt is the name of the file that will be created whit the prediction  

Each row in the eeg_train must be one trial signal (electrodes concatenated) separated by tab with the label in the end. example:  
-26.3971,18.0888,27.1034,-15.5235,...,-60.1944,-31.4051,1  
45.5472,58.7872,-10.9835,-60.4336,...,-28.9022,34.5180,1  
58.3910,19.0882,-41.2055,-41.2985,...,29.7336,70.04784,2  
                .  
                .  
                .  
22.9512,-30.5270,-21.1711,17.9456,...,40.6101,26.8475,2  
-22.8727,-51.1131,-7.8023,40.3250,...,58.7872,-10.9835,2  

Ps.: in the eeg_test must be the label column (can be put '0' in the position if it's unknow)  

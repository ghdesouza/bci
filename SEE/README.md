
For run the SEE approach can be used the command:  
# python3 SEE.py data_file.txt label_file.txt ascending lowcut highcut  

* data_file.txt is the name of the data file that will be used to be predicted  

* label_file.txt is the name of the true label. if the label_file.txt exists, the code will return the found kappa value and if it doesn't exist the code will create this file with the prediction.  

* ascending is a boolean (0 or 1) that point if the sorting is ascending or not  

* lowcut is the low cut value in the bandpass filter (optional)  

* highcut is the high cut value in the bandpass filter (optional)  

Each row in the data_file must be one trial signal separated by comma. example:  
-26.3971,18.0888,27.1034,-15.5235,...,-60.1944,-31.4051  
45.5472,58.7872,-10.9835,-60.4336,...,-28.9022,34.5180  
58.3910,19.0882,-41.2055,-41.2985,...,29.7336,70.04784  
                .  
                .  
                .  
22.9512,-30.5270,-21.1711,17.9456,...,40.6101,26.8475  
-22.8727,-51.1131,-7.8023,40.3250,...,58.7872,-10.9835  

each row in the label_file must be (or will be) the label of each trial. Example:    
1  
1  
2  
1  
.  
.  
.  
2  
2  

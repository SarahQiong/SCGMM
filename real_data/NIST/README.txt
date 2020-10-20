This is the description for extracting features from the NIST dataset


##############################################
# NIST dataset download
##############################################
Directory for dataset download: https://www.nist.gov/srd/nist-special-database-19
download the 2nd edition release on September 2016
download the zip file by_class
read the user guide for the description of the subfoloders
1. the directory 30-39 contains the digits 0-9
2. within each directory, there is a suggested subdirectory for training
3. the hsf_4 is the suggested test directory


#############################################
# Feature extraction
#############################################
1. Train the model on NIST training dataset
	python nn_feature_extractor.py --train
this model gets a classification accuracy of about 98.34% on the training and 97.67% on the test

2. Use the pretrained model to do feature extraction
    
    python nn_feature_extractor.py --evaluate

3. Use extracted 50D features from NIST dataset and perform clustering




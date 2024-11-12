In the dataset folder, each subfolder corresponds to a feature matrix for a patient with the same name as the subfolder.

Within each subfolder, files named 1, 2, 3, ... .mat represent the patient's ictal (seizure period) network matrices, while 
files named feature_1, feature_2, ... contain extracted features from files 1, 2, ... 
Each feature file includes 28 columns representing 4 metrics (mean, variance, maximum, minimum) across 7 frequency bands
 (full, delta, theta, alpha, beta, gamma, HF), with the 29th column indicating the patient's surgical outcome 
(1 for success, 2 for failure) and the 30th column containing the patient's ID, implying s001~s010.
And feature file named the same as patient's ID are the integration of the aforementioned feature files.

all_FE.mat is the feature matric of all these patients after shuffling.

classify_double.m is the MATLAB code for the binary-classified surgical outcomes prediction task, using Support Vector Machine (SVM),
Random Forest (RF), K-nearest neighbors (KNN), and Linear Discriminant Analysis (LDA).
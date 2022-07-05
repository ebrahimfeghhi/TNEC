# TNEC (Tranisent Neural Event Consistency)
Code for computing Transient neural event consistency scores. This package can be used by following the steps outlined below. For any questions,
please email efeghhi@gmail.com.

### Step 1: Setting up the environment. 
Run pip install -r requirements.txt to install the necessary packages. 

### Step 2: Modify setup_data.yaml and run corresponding .py file 

### Step 3: Modify compute_consistency_scores.yaml and run corresponding .py file. 
This step requires two parts. 

First, you should do K-fold cross validation on the training data. After achieving a threshold accuracy across
the folds, compute mean number of epochs across the K-folds (an example of this can be found in saved_models/kfold/examine_kfold.ipynb. 

Second, run N models on the entire training dataset for the number of epochs run above. During this step, it is important to verify that the basline accuracy is high. Once this has been verified, combine predictions from N models. The combined predictions can then be used to generate a network consistency score, or evaluate TNEs that are inconsistent with the reference dataset. An example of this can be found in saved_models/final_models/examine_final_models.ipynb.




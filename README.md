# COCO
This is an implementation of the COCO algorithm proposed in the following NeurIPS 2024 paper:

[1] Sinha, Abhishek and Vaze, Rahul, "Optimal Algorithms for Online Convex Optimization with Adversarial Constraints", Advances in Neural Information Processing Systems, 2024. (preprint available at: https://www.tifr.res.in/~abhishek.sinha/files/COCO_Sinha-Vaze24.pdf).

It trains a neural network with a single hidden layer using the algorithm proposed in the above paper to detect credit card frauds in an online fashion with a highly imbalanced dataset.

The easiest way to replicate the ROC plot reported in the paper is to upload the "colab-code" directory located in the "fraud_detection" branch to your google drive. Then run the notebook "COCO_sim.ipynb" on Google colab. Make sure that you have already downloaded the dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and placed it in the same directory on your Google Drive. 

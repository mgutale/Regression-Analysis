# Regression Analysis on Diamond Dataset

---

Author: [Mo Gutale](https://github.com/mgutale)

<img src="https://media.giphy.com/media/eHvd40K8OfCRRLB3t5/giphy.gif" width="950" height="280" />

---

## Brief Outline 

Businesses often nowadays require to predict prices of products based on a various product attributes and along with other methods in order to ensure that products are priced correctly. This notebook will aim to solve this business problem using the famous diamond dataset. By the end of this notebook, i am expecting to have a model that can be used by business to deploy in a real world scenario. 

## Dataset

This classic dataset contains the prices and other attributes of almost 54,000 diamonds. The source of this dataset can be found [here.](https://www.kaggle.com/datasets/shivam2503/diamonds) 

Features include:

- price in US dollars - Target
- carat weight of the diamond 
- cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- color diamond colour, from J (worst) to D (best)
- clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
- depth (43-79)
- x length in mm (0--10.74)
- y width in mm (0--58.9)
- z depth in mm (0--31.8)

The last three variables represent the dimension of the the particular observation(diamond).

---

## Table of Content

1. Data Load & Quick Look 
2. Explore, Clean & Transform 
3. Feature Selection and Preprocessing
4. Model Training 
5. HyperTunning the Model
6. Feature Importance
7. Prediction on Test & Confidence Interval
8. Save Model 
9. Prediction on Validation set
10. Conclusion 



---



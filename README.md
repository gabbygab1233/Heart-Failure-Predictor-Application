[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgabbygab1233%2FHeart-Failure-Predictor-Application&count_bg=%23231920&title_bg=%23F76767&icon=&icon_color=%23E7E7E7&title=Heart+Failure+Predictor&edge_flat=false)](https://hits.seeyoufarm.com)

# Predicting the Survival of Patients with Heart Failure

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. Four out of 5CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.



<p align="center">
<img src="https://americanaddictioncenters.org/wp-content/uploads/2016/06/heart-icon.png" width="250" height="250" />
</p>
Individuals at risk of CVD may demonstrate raised blood pressure, glucose, and lipids as well as overweight and obesity. These can all be easily measured in primary care facilities. Identifying those at highest risk of CVDs and ensuring they receive appropriate treatment can prevent premature deaths. Access to essential noncommunicable disease medicines and basic health technologies in all primary health care facilities is essential to ensure that those in need receive treatment and counselling.

# Dataset
The dataset contains medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features. https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

### Attributes information:
![](https://i.imgur.com/niqOo77.png)

### Experiment Results:
* **Data Analysis**
    * 5 columns contains outliers this columns are ( creatitinine_phosphokinase, ejection_fraction, platelets, sereum_creatinine, serum_soidum).
    * Imbalanced target class ( I'll used resampling techniques to add more copies of the minority class )
 * **Performance Evaluation**
    * Splitting the dataset by 80 % for training set and 20 % validation set.
 * **Training and Validation**
    * After training and experimenting different algorithms using ensemble models have good accuracy score than linear and nonlinear models.
    * Gradient Boosting Classifier ( 93 % accuracy score )
 * **Fine Tuning**
    * Using {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 1000, 'subsample': 0.7} for Gradient Boosting Classifier improved the accuracy by 1 %.
 * **Performance Results**
    * Validation Score: 97%
    * ROC_AUC Score: 96.9 %
 
 

# Demo
Live demo: https://heartfailure-predictor.herokuapp.com/

Kaggle Kernel:  https://www.kaggle.com/gabbygab/patients-survival-prediction-web-application


![](https://i.imgur.com/Yrn231v.png)


# References
* https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5
* https://github.com/batmanscode/breastcancer-predictor
* https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

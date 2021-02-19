# machine-learning-challenge

>  Create machine learning models capable of classifying candidate exoplanets

## Table of contents
* [General Information](#General-Information)
* [Feature Selection](#Feature-Selection)
* [Compare Models](#Compare-Models)
* [Analysis](#Analysis)
* [Technologies](#technologies)
* [Contact](#Contact)

## General Information
Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.

To help process this data, multiple machine learning models were tested to find the best model which is capable of classifying candidate exoplanets from the raw dataset.

Documentation: [Exoplanet Data Source](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)

Best Model: [KNearestNeighbor](exoplanet_model.sav)


## Feature Selection
Determined the best features by using a heatmap comparing all features and RandomForestClassifier model to find the most important features.

The heatmap shows a correlation between the features. Squares that are more green show features that are highly correlated. Including them would increase their importance and skewing the model.

![HeatMap](Resources/heatmap.png)

The following is the list of features by importance. Features that have err in their name are part of the "Uncertainties Column" in the dataset. This could be considered to be "margin of error" and have been excluded from the feature selection.

![feature_importance](Resources/feature_importance.png)

The final features selected are ```koi_fpflag_co, koi_fpflag_nt, koi_fpflag_ss, koi_model_snr, koi_prad and koi_fpflag_ec``` as they had the highest importance.



## Compare Models

Used GridSearch to hypertune parameters. Parameters used: C = [1, 5, 10]  and kernel = ('linear', 'rbf'). SVC {'C': 50, 'kernel': 'rbf'} was the best parameter with the highest precision identifying False Positives. This model has the following metrics with model accuracy of 0.79
![svc](Resources/svc.png)

Tested Decision Tree Classifier resulted in the following metrics. The model accurancy (0.85) is better with the highest precision identifying False Positives.
![clf](Resources/clf.png)

The Random Tree Forest was tested with the following metrics. This model's accuracy (0.87) is better than the previous models, again with the highest precision identifying False Positives.
![rf](Resources/rf.png)

Finally tested KNearestNeighbors. This resulted in the best model accuracy (0.88) with k=7.
![knn](Resources/knn.png)


## Analysis

SVC model with kernel=rbf and C=50 had the lowest accuracy of 0.79. Next came the Decision Tree Classifier model with an accuracy of 0.84. Both KNearestNeighbor and Random Forest Classifier had higher accuracy scores of 0.88 and 0.87 respectively. While the Random Forest Classifier is better suited for disproportionate data, as in the exoplanet data, the KNearestNeighbor gave a slightly better accuracy overall. The False Positive scores for both models are identical, whereas the Candidate and Confirmed scores are slightly better in the KNearestNeighbor model with k=7. Thus the best model for this data set would be KNearestNeighbor.

The KNearestNeighbors model gives an 88% accuracy in determining candidate exoplanets. The features were selected using RandomForestClassifier to find features with the most importance. The heatmap helped narrow down and remove parameters that were highly correlated so as not to give too much importance to similar features. Finally, by going through the documentation provided on the dataset I eliminated features with 'err' assuming that these probaly refer to "Margin of Error". With the help of an astrophysicist, it would be possible to refine the selection of the most important features and end up with a better model.

## Technologies
* Python
* Pandas
* matplotlib
* numpy
* seaborn
* sklearn
* joblib

## Contact
Created by [@deepavadakan](https://github.com/)
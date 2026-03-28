NASA Wildfire Detection and Prediction Project
Overview
This project uses NASA satellite fire detection data to develop machine learning models for identifying major wildfire days and forecasting fire spread. The pipeline processes large-scale VIIRS/MODIS datasets, balances the data, trains deep learning models, and visualizes results on geographic maps to support wildfire monitoring and early warning.

Contents
Data and Processing
Combines multiple large CSV files of NASA VIIRS/MODIS fire detections.
Filters detections by confidence level to ensure reliability.
Rasterizes fire detections onto a geographic grid (latitude-longitude bins) per day.
Labels each day as "major fire" or "no major fire" based on a percentile threshold of fire hotspot counts to balance classes.
Modeling
Implements a Convolutional Neural Network (CNN) to classify whether a given day has major fire activity.
Uses stratified k-fold cross-validation and class weighting to handle imbalanced data and prevent overfitting.
Employs a ConvLSTM model to learn spatial-temporal patterns and predict fire spread for the following week.
Integrates dropout and early stopping for model regularization and optimized training.
Visualization and Evaluation
Visualizes actual and predicted fire occurrences on world maps using Cartopy.
Reports robust cross-validated metrics including accuracy, precision, recall, and F1-score.
Enables spatial interpretation of wildfire patterns and forecasting accuracy to inform environmental monitoring efforts.
How to Use
Place your NASA VIIRS/MODIS CSV files in the working directory.
Run the comprehensive Python script to preprocess data, train models, and visualize results.
View interactive and static map visualizations for spatial understanding of wildfire events.
Examine printed classification metrics for model performance insights.
Requirements
Python 3.x environment
Libraries: pandas, numpy, scikit-learn, tensorflow, matplotlib, cartopy, tqdm
Sufficient RAM to handle grid size and data volume (adjustable in the script)
References
NASA VIIRS/MODIS satellite fire detection datasets
TensorFlow Keras deep learning framework
Cartopy library for geospatial visualization
Dataset link: https://firms.modaps.eosdis.nasa.gov/active_fire

# Exoplanet-Classification-with-LR-and-DTC
This Python code is for building a classification model using Logistic Regression to predict whether a star has an orbiting exoplanet based on time series data. Here's a summary of the code:

1. The necessary libraries are imported, including pandas, numpy, sklearn modules for train-test split, data preprocessing, classifiers, evaluation metrics, SMOTE for data balancing, and matplotlib/seaborn for visualization.

2. Train and test data are loaded from CSV files.

3. Basic exploratory data analysis is conducted, such as checking the shape of the train and test data, class distribution, and missing values.

4. New features are derived from the time series data (mean, standard deviation, skewness, and kurtosis) and plotted using histograms and boxplots.

5. The data is preprocessed by removing the target variable from the feature sets.

6. SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the training data, ensuring equal representation of both classes (stars with and without exoplanets).

7. The data is standardized using StandardScaler.

8. The training data is split into training and validation sets using train_test_split.

9. A Logistic Regression model is initialized and trained on the resampled and scaled training data.

10. The model is used to predict the target labels for the validation set.

11. A classification report is printed to evaluate the model's performance, including precision, recall, F1-score, and support.

Overall, the code demonstrates the process of data preparation, feature engineering, resampling, model training, and evaluation for a classification problem in astronomy using Logistic Regression.




Also, 

This code performs binary classification on time-series data from the Kepler space telescope, which aims to identify exoplanets (label 1) and non-exoplanets (label 0) based on their brightness variations over time.

The process involves the following steps:

1. Data Preparation:
   - The training and test datasets are read from CSV files into `train_data` and `test_data` DataFrames, respectively.
   - Some basic information about the data, such as the shape and class distribution, is printed.

2. Feature Engineering:
   - Four statistical features, namely mean, standard deviation, skewness, and kurtosis, are computed from the brightness variations for each observation in both training and test datasets.
   - Histograms and box plots are visualized for the computed features.

3. Data Splitting and Resampling:
   - The training dataset is split into training and validation sets to evaluate the model's performance.
   - The training dataset is then resampled using the SMOTE (Synthetic Minority Over-sampling Technique) method to address class imbalance, generating synthetic samples for the minority class (exoplanets).
   - The feature values are scaled using StandardScaler to normalize the data.

4. Model Training and Evaluation:
   - A logistic regression model is initialized, and it is trained on the resampled and scaled training data.
   - The model is used to predict labels for the validation set.
   - The classification report, which includes metrics such as precision, recall, F1-score, and support, is printed to assess the model's performance on the validation set.

Overall, this code aims to build and evaluate a logistic regression classifier for identifying exoplanets in time-series data based on their brightness variations. It uses feature engineering, data resampling, and scaling techniques to improve model performance and address class imbalance in the dataset. The classification report provides insights into the model's ability to distinguish between exoplanets and non-exoplanets and helps in understanding its strengths and weaknesses.

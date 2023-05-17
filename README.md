# Air transportation fare prediction project
- Domenico Alberini (762411)
- Federico Jacopo Baldoni (763691) - Captain
- Vincenzo Rosario Musto (758701)
## Introduction
In this project, a dataset regarding the flights ticket prices in the Indian market is analyzed. The main aim of this project is to create a regression algorithm which can predict the ticket prices, given the information about more than 300.000 flights.

At first, we will work on an Exploratory Data Analysis and we will test some different machine learning algorithms. In the end, we will be working on the hyperparameters tuning of few algorithms and with the feature analysis.


## Exploratory Data Analysis

First of all, we have checked that the dataset did not contain any missing or duplicate value. As we did not have to handle these values, we went on with the next steps.

Then, we have decided to plot a boxplot of our target variable, which is the price of the flights.

As we can see from the picture below, there are some outliers on the right part of the boxplot, we have decided to drop them to remove the noise in the dataset and to improve the performances of the algorithms which we are going to see next.

![Price Boxplot](https://github.com/baldogiovine/763691/blob/main/images/price_boxplot.jpg)


Moreover, we have decided to plot the distribution of the price variable. As we have noticed a gaussian curve for the low prices and another for the high prices, we did think that it can depend on the class of the flight, respectively “Economy” and “Business”. We have decided to plot the distribution of the price variable for each class. As we can see from the plot below, the economy flights are way cheaper than the business flights.

![Price by class](https://github.com/baldogiovine/763691/blob/main/images/price_by_class.jpg)

Then, we wanted to check the price distribution respecting to the number of stops for the flights.

From the boxplot below, it can be inferred that the price of flights with one stop is higher, compared to those with no stops or two or more stops.

![Boxplot price-stops](https://github.com/baldogiovine/763691/blob/main/images/boxplot_price_stops.jpg)

To investigate the reason why, we thought it could depend on the duration of the flights, and as can be seen in the boxplot below, flights with zero stops last significantly less in terms of hours. This also explains why, counterintuitively, flights with no stops have prices significantly lower than those with one stop and two or more stops. In fact, the flights with no stops could regard nearer cities, comparing to the flights with stops.

![Boxplot duration-stop](https://github.com/baldogiovine/763691/blob/main/images/boxplot_duration_stop.jpg)

### Column Transformer

This dataset contains 11 variables.

In particular, there are 3 numerical variables and 8 categorical variables.

To build our regression models we have firstly made the data suitable for this task.

We have decided to build 2 different column transformers, which will be used in the pipelines of each model, even if in the paper we will report only the results from one of it, in order to make it as clean as possible.

The first column transformer (ct) is composed of a OneHotEncoder for all the categorical variables in the dataset, with the exception of the column “stops” which goes through a OrdinalEncoder, as the number of stops for a flight can be considered in a ordinal way. Then, we have used a PowerTransformer for the “duration” column, as the data looked more or less like a skewed gaussian, so we tried to make it more gaussian-like. In the end, we have used a StandardScaler on the “days_left” variables as it farly reminded us a uniform distribution, so we are just standardizing it, in order to standardize it.

The second column transformer (ctc) is basically the same as the first one, but with the exception for the implementation of a custom cyclic for the “departure_time” and “arrival_time” columns, which aim is to make it understandable for the regression models that the times of the day are cyclical, meaning that after late night there is again the early morning. This implementation has been inspired after some works on time series forecasting during another project. The main idea behind it is to order the categories with an ordinal encoder (from early morning to late night), then on both variables sin and cosin are calculated on the value obtained from this formula: (2π*x)/(max_val), where x is the ordinal encoded value of the category, while max_val is the number of unique categories.

![Column transformer](https://github.com/baldogiovine/763691/blob/main/images/column_transformer.jpg)

![Cyclical column transformer](https://github.com/baldogiovine/763691/blob/main/images/column_transformer_cyclical.jpeg)

Moreover, we have tried to perform a box-cox with a PowerTransformer on the target variable (price), but since it is distributed like two different independent functions, performing this kind of manipulations of it, actually gets us in a worse situation so we have decided to keep the variable without any kind of manipulation.

### Correlations

After performing the column transformer, we have decided to check the correlations of all the encoded and manipulated variables with the target (price) variable.

At first, we are checking the correlation, we can see from the picture below that the Economy class is highly negatively correlated with the price, while the Vistara airline is kind of positively correlated with the price.

![Price correlation](https://github.com/baldogiovine/763691/blob/main/images/price_correlation.jpg)

Moreover, we have decided to check the mutual info regression. After some researches, we have discovered that when a lot of categorical variables are present in a dataset, it is usually better to check for the dependency, instead of the classical linear correlation and so, to use algorithms like mutual info regression.

In fact, the mutual information between two random variables is a non-negative value, which measures the dependency between the variables. Dependency suggests that changes in one variable directly affect the other variable. It implies a cause-and-effect relationship. The graph below tells us that there is a high dependency between the duration of the flight and the price.

![Mutual info regression](https://github.com/baldogiovine/763691/blob/main/images/mutual_info_regression.jpg)

### Stratified train-test split

As we have seen in the EDA section, we have seen that the “class” categorical variable values are characterized by way different price values. For this reason, we have decided to stratify the train and test subsets based on the “class” feature. We tried to implement other ways to stratify the split, based on the use of buckets on the price variable but without any improvement.

**### Evaluation Metrics:**

Concerning the metrics used to evaluate the performance of models and to compare them among each other, three metrics have been chosen, the pros and cons of which will be elucidated.

_**R^2**:_

Pro:

It is an intuitive and easy metric to interpret. Even the best known as it is widely used in academic and industrial literature to evaluate model performance.

Cons:

It is not a robust metric and can be influenced by outliers or anomalous data. It does not take into account the complexity of the model, namely the number of variables used.

_**RMSE**:_

Pro:

It is robust to outliers and anomalous data

Cons:

It does not take into account the direction of errors, only their magnitude. Leading in our case to not being able to tell whether there is an under- or over-estimation of the predicted prices.

_**MAPE**:_

Pro:

The metric is very easy to interpret and therefore the most business-friendly. It is the best for communicating business results to those who do not have statistical skills.

Cons:

One limitation is that it may not be the best fit for models where the dependent variable can take on a wide range of values.

### Models

As we can see from the flowchart below, we have implemented different regression models. In particular, we have divided them in linear, ensamble trees and artificial neural networks. Moreover, we have created a requirements.txt to recreate our virtual environment.

![flowchart](https://github.com/baldogiovine/763691/blob/main/images/flights_flowchart.jpg)

### Experimental design

First of all, we have implemented a **linear regression with polynomial features**.

We have thought that there could be the possibility to catch more information by looking for non linear relationships.

The results are incredibly good, considering that the model is a linear regression.

R2:

train 0.9630774982540425

test: 0.9623884603163833

RMSE

train 4346.929687699706

test: 4403.1808362797765

MAPE:

train 0.2243831804557979

test: 0.22643025829910537

We have also tried to perform a SVR but, due to its incapacity to scale on big datasets, we have decided to drop it, due to a very high computational cost. Then, we have dropped a SGD regressor as results were not reliable and it also performed with negative R2.

Then, we went on with the ensemble tree models. We have tried many of them, like ADA boost and Gradient descent boosting, but the only one worth of mention are Random Forest, Hist Gradient Descent Boosting and XGBoost.

Regarding **Random Forest**, it was the best performing among the not-tuned implementations but, performing a grid search on it did not optimize the performances, while the computational costs were prohibitive.

Its results are:

R2:

train 0.9975941370392424

test: 0.9854550804662985

RMSE

train 1109.6154590237916

test: 2738.1776887449373

MAPE:

train 0.02703663120705267

test: 0.06988346428016631

Then, we have implemented **Hist Gradient Descent Boosting**, which is an extremely fast-to-train algorithm, with gave us good results and scales extremely well with big datasets. After tuning the model with a grid search, we got these results:

R2:

train 0.9930944521324994

test: 0.9877702282934792

RMSE

train 1879.9059637774599

test: 2510.8174655599623

MAPE:

train 0.08428840344965698

test: 0.09867852717171964

In the end, we implemented a **XGBoost** model based on DMatrices. We imported xgboost with the native API since we needed a fast-to-train model, while the sklearn implementation do not support the DMatrices. Then, since this version of XGBoost do not rely on sklearn, we could not use the Grid Search as we did for the other models. We implemented Optuna instead, so we could perform the hyperparameter tuning.

The scores for this model are:

R2:

train 0.9928809721587037

test: 0.9876262162062768

RMSE

train 1908.742753633389

test: 2525.557307847127

MAPE:

train 0.08778987942078788

test: 0.10417789082427886

### Artificial Neural Networks

Another implementation to face our regression model is an Artificial Neural Network.

As we could not know a priori which number of layers or neurons we should use in this ANN and the value of these hyperparameters can have a huge influence on the performances, we have decided to tune these and other hyperparameters, as the learning rate, through Keras Tuner, which helps us to easily tune the ANN model based on Keras and TensorFlow.

In particular, we let the tuner to choose between 2 to 6 hidden layers, with a number of nodes for each layer (after the first one) from 32 to 544, with a step of 256, in order to try a number of nodes in each iteration which is different enough from the other iterations. Moreover, 3 different learning rates are tuned.

To make the process faster, only 100 random combinations are tried. But, as Keras Tuner saves the information of the tuning in a subfolder, during the next runs of the project’s code, the tuning will not be done, as Keras Tuner can retrieve the information from that folder. In this way, the code is smoother to run. But, in the case the reader wants to try the tuning on their machine, it is possible to delete the folder or to change the project name in the tuner code.

We have decided to use ReLU as activation function for the hidden layers as it is not computationally expensive while it still provides non-linearity while, for the output layer we have decided to use a linear activation function, as we are dealing with a regression problem, where the values are continuous and it avoids the need of further transformations to interpret the results.

We can therefore plot the resulting ANN model:

![ANN](https://github.com/baldogiovine/763691/blob/main/images/model_ANN_ctc.jpg)

As the ANN captures the non-linear relationships inside the dataset, with great performances and with the possibility to store the training data, we have decided to keep this model in our project.

The scores for ANN are:

R2:

Train: 0.9801484515832133

test: 0.9772817569712573

RMSE

Train: 3187.3829800135118

test: 3422.105044720033

MAPE:

Train: 0.11757862594594162

test: 0.1259337740085407

## Results

Although XGBoost and HistGrandientBoostRegressor performances are extremely similar, we decided to consider the second one (HistGrandientBoostRegressor) the best one since it was cross-validated, and its performances are more reliable. It's also notable that, being HistGrandientBoostRegressor a sklearn model, eventual further implementations are often easier. Moreover, we discarded the ANN as it is performing a little worse, while it is more computationally expensive and less interpretable.

Therefore, we want to deeper understand how this model works so we analyzed the importance of the features. To do this, we implemented Shap. The fundamental idea behind SHAP is based on game theory and the concept of Shapley values. Shapley values are a concept from cooperative game theory that assigns a value to each player in a game, based on their contribution. In the context of SHAP, the players are the features of a machine learning model, and the value is the impact of each feature on the model's predictions.

Looking at shap values we can see that, how we expected, the most important feature is class. In fact, if we go back looking at EDA, it is possible to notice how the target variable (price) is explained by class feature. Class explains those two cusps: the first is related to economy category and the second one by business category. Value of one category does not affect values of the other category: they seem to be two separated functions.

![Shap](https://github.com/baldogiovine/763691/blob/main/images/shap1.jpg)

Then, we have plotted the mean of the absolute shap values of the features, in order to understand the absolute impact of each feature on the target variable (price).

![Shap bars](https://github.com/baldogiovine/763691/blob/main/images/shap2.jpg)

To confirm what we just said, here you can see how optimazing a model with only the class feature still leads to a pretty decent result with:

R2:

train 0.8810495906985512

test: 0.8812741419097986

RMSE

train 7802.252948332393

test: 7823.095366557577

MAPE:

train 0.4390154575150844

test: 0.4373283906939269

We can infer that using also all the other features allow us to have grater performances that almost reach an R2 of 0.99, but that most of the information inside the model is provided by class feature.

## Conclusions

This work helped us to improve our practical skills in machine learning. It was clear to us that this task was solvable in many different ways with great results. Our implementations of a cyclic encoder, grid search, Optuna, Keras Tuner and Shap made us more conscious of the most effective methods in the machine learning world. In the end, the task was solved with a great R2 value (0.987) with a cross validation. Due to time limits and limited computing capacity, we had to work with a 3-fold cross validation. Moreover, due to time constraints, we did not implement a cross validation on XGBoost, while we could not properly comment the code on the notebook. We hope we can improve the comments in the next days if it is allowed.

In the end, we want to thank Professors Italiano and Torre for the useful guidance during the semester.

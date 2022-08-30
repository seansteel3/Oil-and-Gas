# Oil-and-Gas
Can classical and modern regression analysis reveal what the major drivers of global oil and gasoline prices are?

For a complete set of visualizations check out the Oil & Gas PDF.

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Summary](#summary)
* [Introduction](#introduction)
  * [Run With](#run-with)
* [Data Sources](#data-sources)
* [A Journey to MARS](#a-journey-to-mars)


<!-- Summary -->
## Summary

With rising costs of gasoline and crude oil prices nearing all-time highs, understanding what drives this market is a question well suited for regression analysis. Specifically, the aim of this project was to uncover general indicators and economic predictors of oil price using and contrasting multiple regression (MR) and Multivariate Adaptative Regression Splines (MARS). Both MR and MARS produce similar models “out of the box” but are overfit. MR and MARS both fail to capture the same large drop in oil price, but MR also fails to capture the general trend of oil price after the drop, while MARS characterizes this trend well. To construct a model with more accurate predictions in the out-sample, two methods were applied to MR. A back-selection procedure, removing one term at a time until no further improvement in mean squared error, was applied to the full MR model. Concurrently MARS was used in a feature selection setting, running the algorithm on 200 80-20 train/discard splits, to develop a competing MR model. Of the 201 MARS runs (200 splits and 1 full model) features occurring in less than 25% of models were discarded, and the remaining terms were formed into a MR model. The MARS feature selection MR model outperformed the back selection MR model for both oil price prediction and gas price prediction (MARS MR MSE = 42.54, back selection MR MSE = 66.32). The final model showed the most highly correlated features to oil price are Running Deficit, US Industrial Production, Global Industrial Production, US Retail Sales, Global CPI, and Global Core CPI. Lastly, this final model’s price predictions, along with other candidates for comparison, were piped into a MR model to predict US average gasoline price (gas price = oil price + refinery profit margins). Overall absolute error of the oil price prediction model = 7.2% and overall absolute error of the gasoline price prediction model = 4.2%, across the whole dataset.


<!-- Introduction -->
## Introduction
In the current political climate of 2022, all sides have an idea of why oil and gasoline prices have risen to current highs. Regardless of political affiliation, much of the data required to understand the current global pricing is publicly available and relatively straight forward to analyze with modern analytics tools. At least to develop a high-level understanding of the major factors at play.

This project aims to uncover high level drivers of oil and gasoline prices, as well as to explore MARS and Multiple Regression in this context as data mining applications as well as predictive models.

### Run With
Python 3.9.7
 * pandas
 * numpy
 * statsmodels
 * sklearn
 * pyearth

<!-- Data Sources -->
## Data Sources
US oil industry data and gasoline price data was collected from the EIA’s short term energy outlook, and drilling productivity report data sets. Data pertaining to total oil production and consumption came from the Bureau of Land Management was validated from related tables in the EIA’s datasets and the Natural Resources Revenue Data. Yahoo finance supplied average monthly oil prices. Global economic indicators were collated from the World Bank’s Global Economic Monitor and relevant US metrics were validated from data publicly available from either the White House or the Bureau of Labor Statistics.

All data sources that could not be correlated or otherwise validated by at least two sources (except for oil and gasoline price data) was discarded. All included monthly data was transformed and collated into one summary oil statistics table, while data of interest that could only be found at yearly intervals was collated into a separate yearly data table. Production and consumption data was transformed from average mega barrels (or gallons) of oil (or gas) per month to total consumed barrels (or gallons) per month.
One custom variable of importance, Running Deficit, was created to estimate general supply and demand by subtracting monthly global oil production from monthly global oil consumption to keep a running tally of available supply. Note this metric erroneously begins at 0 in Jan 1997 due to data limitations, and negative numbers do not imply inadequate or non-existing global supply 
See Data Sources excel file for relevant links.


This project aims to uncover major correlates of historical oil prices (and therefore gasoline prices) to understand the current market dynamics in 2022. Additionally, due to the large amount of data and data sources available, this project also aims to experiment with, explore, and apply a machine learning algorithm known as Multivariate Adaptive Regression Splines (MARS) in both a model construction setting and a feature selection setting.



<!-- A Journey to MARS -->
## A Journey to MARS

MARS is an adaptive rapid regression algorithm developed to rapidly identify linear and non-linear interactions between variables without the issues of long training epocs or data loss inherent to other machine learning methods. The crux of MARS relies on linear regression splines. In general, a spline regression is generated by fitting a series of piecewise equations where breaks occur at various “knots.”

The MARS algorithm automatically detects knot locations and higher order interactions between variables when suitable. MARS generally requires minimal data preparation and produces models that are nearly as interpretable as classic multiple regression models. 
Formally MARS uses the positive portion of what are called hinge functions to separate the data at previously mentioned spline knots.


These hinge functions are applied adaptively throughout the entire data set, and additively summed together to give our final model. In mathematical notation our final model has the same form as multiple regression:

![Screenshot 2022-08-30 122702](https://user-images.githubusercontent.com/67161057/187490802-026bfc4a-470f-42cf-8d7d-e45f8e76bfd4.png)

Where X is the data,  β_0 is a regression constant, M is the number of hinge functions used, h_m is a particular hinge function, and β_m is the coefficient for a particular hinge function.

To create the above model MARS proceeds forward as follows:

1. Regress a constant (β_0) on all the data
2. Generate all possible knots for all possible hinge functions across all variables 
3. Select a hinge function by multiplying it to any already selected term in the model 
4. Select the “best” candidate in step 3 by least squares
5. Repeat steps 2-4 until a set maximum number of variables reached or until a decrease in squared error reaches some threshold.

Generally, by the end of the last step the model is overfit to the data. Therefore, MARS then applies generalized cross validation (GCV) in a backwards deletion process, removing one term at a time until we have minimized the generalized cross validation metric.

![Screenshot 2022-08-30 122757](https://user-images.githubusercontent.com/67161057/187490510-5d1878f6-3efd-4e0d-a5b2-7633a5278c32.png)

Where RSS is model residual sum of squares, N is number of data points, and pe is effective number of parameters.





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
* [Results](#results)
* [References](#references)


<!-- Summary -->
## Summary

With rising costs of gasoline and crude oil prices nearing all-time highs, understanding what drives this market is a question well suited for regression analysis. Specifically, the aim of this project was to uncover general indicators and economic predictors of oil price using and contrasting multiple regression (MR) and Multivariate Adaptative Regression Splines (MARS). Both MR and MARS produce similar models “out of the box” but are overfit. In the out sample data, MR and MARS both fail to capture the same large drop in oil price, but MR also fails to capture the general trend of oil price after the drop, while MARS characterizes this trend well. To construct a model with more accurate predictions in the out-sample, two methods were applied to MR. A back-selection procedure, removing one term at a time until no further improvement in mean squared error, was applied to the full MR model. Concurrently MARS was used in a feature selection setting, running the algorithm on 200 80-20 train/discard splits, to develop a competing MR model. Of the 201 MARS runs (200 splits and 1 full model) features occurring in less than 25% of models were discarded, and the remaining terms were formed into a MR model. The MARS feature selection MR model outperformed the back selection MR model for both oil price prediction and gas price prediction (MARS MR MSE = 42.54, back selection MR MSE = 66.32). The final model showed the most highly correlated features to oil price are Running Deficit, US Industrial Production, Global Industrial Production, US Retail Sales, Global CPI, and Global Core CPI. Lastly, this final model’s price predictions, along with other candidates for comparison, were piped into a MR model to predict US average gasoline price (gas price = oil price + refinery profit margins). The best model's overall absolute error of the oil price prediction model = 7.2% and overall absolute error of the gasoline price prediction model = 4.2%, across the whole dataset.


<!-- Introduction -->
## Introduction
In the current political climate of 2022, all sides have an idea of why oil and gasoline prices have risen to current highs. Regardless of political affiliation, much of the data required to understand the current global pricing is publicly available and relatively straight forward to analyze with modern analytics tools, at least to develop a high-level understanding of the major factors at play.

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
US oil industry data and gasoline price data was collected from the EIA’s short term energy outlook and drilling productivity report data sets. Data pertaining to total oil production and consumption came from the Bureau of Land Management, and was validated from relevant tables in the EIA’s datasets and the Natural Resources Revenue Data query tool. Yahoo finance supplied average monthly oil prices. Global economic indicators were collated from the World Bank’s Global Economic Monitor and relevant US metrics were validated from data publicly available from either the White House or the Bureau of Labor Statistics.

All data sources that could not be correlated or otherwise validated by at least two sources (except for oil and gasoline price data) were discarded. All included monthly data was transformed and collated into one summary oil statistics table, while data of interest that could only be found at yearly intervals was collated into a separate yearly data table. Production and consumption data was transformed from average mega barrels (or gallons) per month per day of oil (or gas) to total consumed barrels (or gallons) per month.

One custom variable of importance, Running Deficit, was created to estimate general supply and demand by subtracting monthly global oil production from monthly global oil consumption to keep a running tally of available supply. Note this metric erroneously begins at 0 in Jan 1997 due to data limitations, and negative numbers do not imply inadequate or non-existing global supply.

See Data Sources excel file for relevant links.

<!-- A Journey to MARS -->
## A Journey to MARS

MARS is an adaptive regression algorithm developed to rapidly identify linear and non-linear interactions between variables without the issues of long training times, and interpretability/complexity trade-offs inherent to other machine learning methods. The crux of MARS relies on linear regression splines. In general, a spline regression is generated by fitting a series of piecewise equations where breaks occur at various “knots.”

The MARS algorithm automatically detects knot locations and higher order interactions between variables when suitable.
Formally MARS uses the positive portion of what are called hinge functions to separate the data at previously mentioned spline knots.

![image](https://user-images.githubusercontent.com/67161057/187493445-ad0643c1-f713-4139-af6e-6ec520745caf.png)

These hinge functions are applied adaptively throughout the entire data set, and additively summed together to give our final model. In mathematical notation our final model has the same form as multiple regression:

![Screenshot 2022-08-30 122702](https://user-images.githubusercontent.com/67161057/187490802-026bfc4a-470f-42cf-8d7d-e45f8e76bfd4.png)

Where X is the data,  β_0 is a regression constant, M is the number of hinge functions used, h_m is a particular hinge function, and β_m is the coefficient for a particular hinge function.

To create the above model MARS proceeds forward as follows:

1. Regress a constant (β_0) on all the data
2. Generate all possible knots for all possible hinge functions across all variables 
3. Select a hinge function by multiplying it to any already incorporated term in the model 
4. Select the “best” candidate in step 3 by least squares
5. Repeat steps 2-4 until a set maximum number of variables reached or until a decrease in squared error reaches some threshold.

Generally, by the end of the last step the model is overfit to the data. Therefore, MARS then applies generalized cross validation (GCV) in a backwards deletion process, removing one term at a time until we have minimized the generalized cross validation metric.

![Screenshot 2022-08-30 122757](https://user-images.githubusercontent.com/67161057/187490510-5d1878f6-3efd-4e0d-a5b2-7633a5278c32.png)

Where RSS is model residual sum of squares, N is number of data points, and pe is effective number of parameters.

MARS generally requires minimal data preparation and produces models that are nearly as interpretable as classic multiple regression models, with additional insights into higher order interactions and non-linearities.

<!-- Results -->
## Results

### Initial Data Exploration

The initial pass through the data (slides 1-7 in the PDF) aimed to explore the dataset and uncovered a few interesting points. For instance, between January 1997 and May 2022 there is no clear correlation between new oil rig construction and oil production within the US (P = 0.882). Additionally, drilling on public lands, and associated drilling permits granted by the president, are in fact negatively correlated to US oil production (R^2 = 0.82, P = 0.000). Additionally, within this dataset, political party of the president does not have a clear correlation with the number of rigs produced in a year when adjusted for possible delayed effects from the previous administration (P = 0.682). Lastly, US oil production has no correlation to global oil price (P = 0.9), or US average gasoline price (P = 0.083) at any time window between 1997-2022 (P-values for whole dataset). 

None of these points are extremely surprising since oil rig construction does not necessitate productive rigs, nor does it require that all rigs are created equal. For instance, in this dataset one rig producing 3 barrels a month would be marked equivalent to a rig producing 2,000 barrels a month. Further, public land permits and associated drilling is not always productive, and only accounts for about 10-15% of all oil produced in the US. Finally, while the US is the World's largest individual producer today, it only accounts for less than 15% of total oil production (historically substanitally less than 15%), and only produces about 2/3 of its own consumption requirements. Therefore, it is unsurprising that the US alone could affect the supply side enough to control global oil or gasoline prices.

However, gasoline prices are highly correlated to oil prices (R^2 = 0.933, P = 0.000) and refinery profit margins (R^2 = 0.232, P = 0.000). This fact is also not surprising given gasoline is refined from oil, and the cost as well as profit margin of the refinery process would be transferred to the consumer.

*All P-values have been corrected for heteroskedasticity

### Predictive Model Construction

To assess what variables may be most impactful, or at least most correlated to oil change prices, several predictive models were constructed. The first pass began with a full multiple regression (MR) model and a MARS “out of the box” model. Data used for model construction only included months 50-247, where all variables had complete coverage. Training data (in sample) was subset to months 50-200 and test (out sample) 201-247.

Both methods produced similar models in sample, and both failed to capture a large drop around month 215 in the out sample. However, MARS continued to capture the trend in the out sample after month 215, while MR did not (MARS MSE = 924, MR MSE = 1481). 

To improve out sample prediction MARS was run in a feature selection setting. The training data was subset into 200 80/20 random train/discard sets and one full data set. MARS was run over these 201 datasets and the frequency of each feature computed. Any features not included in at least 25% of the models were discarded, and the remaining features were run through a grid search to produce a MR model. Using MARS in this fashion narrowed down the possible number of models from 4.1 million to ~4000. 

After running the 4000 candidate MR models, the top 1000 (by MSE on the out-sample) were given an additional autoregressive term. This term was constructed by regressing the previous 7 months’ worth of oil price data on itself to predict the current month. 7 Months was chosen arbitrarily. Adding the autoregressive term had the desired effect of stabilizing prediction accuracy, while not dominating the model predictions. 

These 1000 models were then run through another MR model for gasoline price (Gas Price = Oil Price + Refinery Profit Margins). The out sample MSE for each model (oil price and gas price) were computed, and each respectively normalized between 0 and 1. The model with the best global rank MSE (normalized MSE for oil + normalized MSE for gas) was chosen as the best model. 

The top model by global rank MSE included the features:
1. Running Deficit
2. US Industrial Production
3. Global Industrial Production
4. US Retail Sales
5. Global CPI
6. Global Core CPI
7. Autoregressive Term (Auto)

To contrast building a MR model with MARS, a standard back selection procedure was implemented on the full initial MR model. The procedure removed one term from the full model, checked the MSE, and permanently deleted the term with the worst improvement in MSE. This was continued until no MSE improvement was detected. 
The final model included the features:
1. Running Deficit
2. US CPI
3. US Industrial Production
4. US Retail Sales
5. Global Unemployment Rate
6. Global Retail Sales
7. Global CPI

This model was then given and auto regressive term and run through the gasoline MR model. 

The MARS feature selection model outperformed the back selection model (MARS Oil Price MSE = 42.54 Back Selection MSE = 65.32 and MARS Gas Price MSE = 0.015 Back Selection Gas Price MSE = 0.02). However, a critical issue remains with these model assessments. Specifically, both MSE’s throughout both processes were computed on the out sample. An important future direction is to run both models on truly unseen data when the World Bank’s datasets are updated.

Final Models (prediction in red, actual in blue):

![Screenshot 2022-08-30 133504](https://user-images.githubusercontent.com/67161057/187505498-2e8b7e7a-c34d-46f2-909c-5ff4bd9b67d8.png)

![Screenshot 2022-08-30 133520](https://user-images.githubusercontent.com/67161057/187505526-a07e48ca-6d59-44c0-9ba0-7c15f3354dc3.png)

![Screenshot 2022-08-30 133137](https://user-images.githubusercontent.com/67161057/187505540-0d112e8f-da5f-459c-a265-6a7c7cfc4b16.png)

![back_sel_gas_auto](https://user-images.githubusercontent.com/67161057/187532951-7ae31a01-48c4-47ea-80db-2eefcb95546e.png)


Remaining visual results in the PDF slideshow

<!-- References -->
## References

* ESLII – Advanced Machine Learning
* Jerome H. Friedman. "Multivariate Adaptive Regression Splines." The Annals of Statistics, 19(1) 1-67 March, 1991
* Remaining sources within the data sources excel file

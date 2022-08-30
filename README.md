# Oil-and-Gas
Can classical and modern regression analysis reveal what the major drivers of global oil and gasoline prices are?

For a complete set of visualizations check out the Oil & Gas PDF.

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Summary](#summary)
* [Introduction](#introduction)
  * [Run With](#run-with)
* [Data Sources](#data-sources)

<!-- Summary -->
## Summary

With rising costs of gasoline and crude oil prices nearing all-time highs, understanding what drives this market is a question well suited for regression analysis. Specifically, the aim of this project was to uncover general indicators and economic predictors of oil price using and contrasting multiple regression (MR) and Multivariate Adaptative Regression Splines (MARS). Both MR and MARS produce similar models “out of the box” but are overfit. MR and MARS both fail to capture the same large drop in oil price, but MR also fails to capture the general trend of oil price after the drop, while MARS characterizes this trend well. To construct a model with more accurate predictions in the out-sample, two methods were applied to MR. A back-selection procedure, removing one term at a time until no further improvement in mean squared error, was applied to the full MR model. Concurrently MARS was used in a feature selection setting, running the algorithm on 200 80-20 train/discard splits, to develop a competing MR model. Of the 201 MARS runs (200 splits and 1 full model) features occurring in less than 25% of models were discarded, and the remaining terms were formed into a MR model. The MARS feature selection MR model outperformed the back selection MR model for both oil price prediction and gas price prediction (MARS MR MSE = 42.54, back selection MR MSE = 66.32). The final model showed the most highly correlated features to oil price are Running Deficit, US Industrial Production, Global Industrial Production, US Retail Sales, Global CPI, and Global Core CPI. Lastly, this final model’s price predictions, along with other candidates for comparison, were piped into a MR model to predict US average gasoline price (gas price = oil price + refinery profit margins). Overall absolute error of the oil price prediction model = 7.2% and overall absolute error of the gasoline price prediction model = 4.2%, across the whole dataset.


<!-- Introduction -->
## Introduction
In the current political climate of 2022, all sides have an idea of why oil and gasoline prices have risen to current highs. Regardless of political affiliation, much of the data required to understand the current global pricing is publicly available and relatively straight forward to analyze with modern analytics tools. At least to develop a high-level understanding of the major factors at play.

<!-- Data Sources -->
## Data Sources
US oil industry data and gasoline price data was collected from the EIA’s short term energy outlook, and drilling productivity report data sets. Data pertaining to total oil production and consumption came from the Bureau of Land Management was validated from related tables in the EIA’s datasets and the Natural Resources Revenue Data. Yahoo finance supplied average monthly oil prices. Global economic indicators were collated from the World Bank’s Global Economic Monitor and relevant US metrics were validated from data publicly available from either the White House or the Bureau of Labor Statistics.
All data sources that could not be correlated or otherwise validated by at least two sources (except for oil and gasoline price data) was discarded. All included monthly data was transformed and collated into one summary oil statistics table, while data of interest that could only be found at yearly intervals was collated into a separate yearly data table. Production and consumption data was transformed from average mega barrels (or gallons) of oil (or gas) per month to total consumed barrels (or gallons) per month.
One custom variable of importance, Running Deficit, was created to estimate general supply and demand by subtracting monthly global oil production from monthly global oil consumption to keep a running tally of available supply. Note this metric erroneously begins at 0 in Jan 1997 due to data limitations, and negative numbers do not imply inadequate or non-existing global supply 
See Data Sources excel file for relevant links.


This project aims to uncover major correlates of historical oil prices (and therefore gasoline prices) to understand the current market dynamics in 2022. Additionally, due to the large amount of data and data sources available, this project also aims to experiment with, explore, and apply a machine learning algorithm known as Multivariate Adaptive Regression Splines (MARS) in both a model construction setting and a feature selection setting.



### Run With
Python 3.10




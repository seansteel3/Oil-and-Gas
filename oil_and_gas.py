# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:38:50 2022


@author: SeanSteele
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tqdm
import pyearth
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import itertools

os.chdir('C:\\Users\\SeanSteele\\Desktop\\Oil')

oil2 = pd.read_csv('full_oil_sum.csv')
month_no = []
for i in range(len(oil2)):
    month_no.append(i+1)
oil2['month_no'] = month_no

oil2['refiner_margin'] = oil2['refiner_margin']/100

oil_yr = pd.read_csv('oil_sum_yr.csv')
year_num = []
for i in range(len(oil_yr)):
    year_num.append(i+1)
oil_yr['year_num'] = year_num


#%%
def auto_reg(df, metric, n_days, formula):
    df2 = pd.DataFrame(df[metric])
    df2['month_no'] = df['month_no']
    preds = [np.nan]*n_days
    end = n_days
    for i in range((len(df)) - end):
        auto_reg_df = df2.iloc[i:end,:]
        reg = smf.ols(formula=formula, data=auto_reg_df).fit()
        preds.append(reg.predict(df2['month_no'][df2.month_no == end+1]))
        end +=1
    final_list = preds[0:n_days]
    for i in range((n_days),len(preds)): 
        final_list.append(preds[i].values[0])
    df['auto'] = final_list
    return df

def sm_reg(data, xvars, yvar):
    X = data[xvars]
    X = sm.add_constant(X)
    Y = data[yvar]
    reg = sm.OLS(Y,X)
    reg = reg.fit(cov_type='HC3')
    return reg

def Find_Pattern(Text, Pattern, numOfPattern):
    in_thisstring = 0 #ensure no double counting if shows up with multiple basis functions
    for each in range(0, len(Text)-len(Pattern)+1): 
       if Text[each:each+len(Pattern)] == Pattern and in_thisstring == 0:
           numOfPattern += 1
           in_thisstring += 1
    return numOfPattern

def mse_2reg(train, test, xvars1, yvar1, xvars2, yvar2):
    #oil regression
    reg1 = sm_reg(train, xvars1, yvar1)
    x_pred = test[xvars1]
    x_pred = sm.add_constant(x_pred)
    pred1 = reg1.predict(x_pred)
    test['oil_pred'] = pred1
    #oil -> gas regression
    reg2 = sm_reg(train, ['oil_price', xvars2], yvar2)
    x_pred2 = test[['oil_pred', xvars2]]
    x_pred2 = sm.add_constant(x_pred2)
    pred2 = reg2.predict(x_pred2)
    mse = np.mean((pred2 - test.gas_raw)**2)
    return mse

def back2model_sel(train, test, x_vars1, yvar1, x_vars2, yvar2):
    #init end point
    max_pop = len(x_vars1)
    pop_index = 0
    max_iter = 100
    iteration = 0
    while pop_index  < max_pop and iteration < max_iter:
        #fit MSE on current full model
        mse1 = mse_2reg(train, test, x_vars1, yvar1, x_vars2, yvar2)
        #drop variable
        x_varsnew = x_vars1[:pop_index] + x_vars1[pop_index + 1:]
        #compute new mse
        mse2 = mse_2reg(train, test, x_varsnew, yvar1, x_vars2, yvar2)
        #if mse1 > mse2 drop variable and restart, if not move on to next variable
        if mse1 > mse2:
            x_vars1 = x_varsnew
            #reset counters
            max_pop = len(x_vars1)
            pop_index = 0
        else:
            pop_index += 1
        iteration += 1
        print (iteration)
    return [x_vars1, mse1]

def mse_reg(train, xvars, test):
    X = train[xvars]
    X = sm.add_constant(X)
    Y = train['oil_price']
    reg = sm.OLS(Y,X)
    reg = reg.fit()
    x_pred = test[xvars]
    x_pred = sm.add_constant(x_pred)
    pred = reg.predict(x_pred)
    mse = np.mean((pred - test.oil_price)**2)
    return mse

def backmodel_sel(train, test, x_vars):
    #init end point
    max_pop = len(x_vars)
    pop_index = 0
    while pop_index  < max_pop:
        #fit MSE on current full model
        mse1 = mse_reg(train, x_vars, test)
        #drop variable
        x_varsnew = x_vars[:pop_index] + x_vars[pop_index + 1:]
        #compute new mse
        mse2 = mse_reg(train, x_varsnew, test)
        #if mse1 > mse2 drop variable and restart, if not move on to next variable
        if mse1 > mse2:
            x_vars = x_varsnew
            #reset counters
            max_pop = len(x_vars)
            pop_index = 0
        else:
            pop_index += 1
    return [x_vars, mse1]

def minmax_scale(df, column):
    scaler = MinMaxScaler()
    scaler.fit(df[column].values.reshape(-1,1))
    return scaler.transform(df[column].values.reshape(-1,1))

#%%
'''
Full model with MARS

'''
#%% full mars predictions
##FULL MARS
mars_full = pyearth.Earth(allow_linear=True)
#drop NA, non-numerics, duplicate info columns, and cheater columns (gas margin + refiner cost = gas price)
x_full = oil2.iloc[:,oil2.columns != 'gas_raw']
x_full = x_full.iloc[:, x_full.columns != 'date']
x_full = x_full.iloc[:, x_full.columns != 'pres']
x_full = x_full.iloc[:, x_full.columns != 'refiner_cost']
x_full = x_full.iloc[:, x_full.columns != 'refiner_sale']
x_full = x_full.iloc[:, x_full.columns != 'gas_margin']
x_full = x_full.iloc[:, x_full.columns != 'gas_adj']
x_full = x_full.iloc[:, x_full.columns != 'running_def_Mbbl']
x_full = x_full.iloc[:, x_full.columns != 'us_prod_Mbbld']
x_full = x_full.iloc[:, x_full.columns != 'opec_prod_Mbbld']
x_full = x_full.iloc[:, x_full.columns != 'glob_prod_Mbbld']
x_full = x_full.iloc[:, x_full.columns != 'us_con_Mbbld']
x_full = x_full.iloc[:, x_full.columns != 'glob_con_Mbbld']
x_full = x_full.iloc[:, x_full.columns != 'glob_deficit_Mbbld']
x_full = x_full.iloc[:, x_full.columns != 'oil_price']
x_full = x_full.iloc[:, x_full.columns != 'refiner_margin']
x_final = x_full.dropna()

y_full = oil2['oil_price']
y_final = y_full.iloc[49:246]

mars_full.fit(x_final, y_final)

full_sum = mars_full.summary()

mars_full.mse_

y_new = mars_full.predict(x_final)

plt.plot(x_final.month_no, y_final)
plt.plot(x_final.month_no, y_new, color='red')

#out of sample drop last 45 days and select significant columns
y_train = y_full.iloc[49:200]
x_train = x_full.iloc[49:200,:]
x_test = x_full.iloc[201:246]

mars_train = pyearth.Earth()
mars_train.fit(x_train, y_train)
train_sum = mars_train.summary()

y_pred = mars_train.predict(x_test)

x_values = x_test[x_test.month_no > 201].month_no
y_real = y_final.loc[201:]


plt.plot(x_values, y_real)
plt.plot(x_values, y_pred, color='red')

#%%
'''
All possible Models space
'''
#%%
#set up autoreg
oil2 =  auto_reg(oil2, 'oil_price', 7, "oil_price ~ month_no")
data = oil2.dropna()
#split train test data
train = data[data.month_no < 200]
test = data[data.month_no > 200]

#all un-diff variables    
xvar_all = ['us_prod','opec_prod','glob_prod','us_con','glob_con','glob_deficit',
        'running_def','republican','new_rigs','us_cpi','us_core_cpi','us_indust_prod',
        'us_retail_sales','oced_gdp','core_cpi','indust_prod','unem_rate','retail_sales',
        'world_cpi', 'auto']

all_combos = []
for c in range(len(xvar_all) + 1):
    all_combos += (list(itertools.combinations(xvar_all, c)))
    
#convert tuples to lists
xlist = []
for i in range(len(all_combos)):
    xlist.append(list(all_combos[i]))


#loop over all models and compute mse - Brute force: too long just ex code
batch_start = 900000
batch_end = 1048575
#set up dataframe and lists to hold results
model = []
out_mse = []
for i in tqdm.tqdm(range(batch_start, batch_end)): #tqdm to look fancy!
    model.append(xlist[i])
    out_mse.append(mse_reg(train,xlist[i], test))


results = pd.DataFrame()
results['mse'] = out_mse
results['model'] = model

results_concat = pd.read_csv('brute_force.csv')
results_concat = results_concat.append(results)
#write results to csv (save in batches)
results_concat.to_csv('brute_force.csv', index = False)

#take all models under 70 MSE
consider_models = results_concat[results_concat['mse'] < 70]
#remove auto from all models and reformat models
not_auto = []
model = []

for i in range(len(consider_models)):
    mod = [j for j in consider_models['model'].iloc[i].split(', ') if j != 'auto']
    mod_orig = [j for j in consider_models['model'].iloc[i].split(', ')]
    not_auto.append(mod)  
    model.append(mod_orig)
consider_models['not_auto'] = not_auto
consider_models['model'] = model

#compute MSE without auto, with auto through gas, without auto through gas
nonauto_mse = []
nonauto_mse_gas = []
auto_mse_gas = []
for i in tqdm.tqdm(range(1,len(consider_models))): #tqdm to look fancy!
    nonauto_mse.append(mse_reg(train,consider_models['not_auto'].iloc[i], test))
    nonauto_mse_gas.append(mse_2reg(train,test, consider_models['not_auto'].iloc[i], 'oil_price','refiner_margin','gas_raw'))
    auto_mse_gas.append(mse_2reg(train,test, consider_models['model'].iloc[i], 'oil_price','refiner_margin','gas_raw'))

consider_models = consider_models.iloc[1:,:]
consider_models['nonauto_mse'] = nonauto_mse
consider_models['nonauto_mse_gas'] = nonauto_mse_gas
consider_models['auto_mse_gas'] = auto_mse_gas

#Normalize MSEs, add auto mses as crital, non-auto as extra, all as total
consider_models['auto_oil_mse_scale'] = minmax_scale(consider_models,'mse')
consider_models['noauto_oil_mse_scale'] = minmax_scale(consider_models,'nonauto_mse')
consider_models['auto_gas_mse_scale'] = minmax_scale(consider_models,'auto_mse_gas')
consider_models['noauto_gas_mse_scale'] = minmax_scale(consider_models,'nonauto_mse_gas')

consider_models['critial_mse'] = consider_models['auto_oil_mse_scale'] + consider_models['auto_gas_mse_scale']
consider_models['non_critial_mse'] = consider_models['noauto_oil_mse_scale'] + consider_models['noauto_gas_mse_scale']

consider_models['total_mse'] = consider_models['critial_mse'] + consider_models['non_critial_mse']

consider_models.to_csv('brute_force_consider.csv', index = False)

#####
consider_models = pd.read_csv('brute_force_consider.csv')
consider_models['gas_mse'] = consider_models['auto_gas_mse_scale'] + consider_models['noauto_gas_mse_scale']
consider_models['oil_mse'] = consider_models['auto_oil_mse_scale'] + consider_models['noauto_oil_mse_scale']
#%%
'''
MARS term frequency exploration Dimentionality reduction
'''

#%%
#init new mars object
mars = pyearth.Earth(allow_linear=True)
#init results container
results = []
#run once over whole exploration data
mars.fit(x_final, y_final)
results.append(mars.summary())

for i in range(200):
    mars = pyearth.Earth(allow_linear=True)
    #80-20 random train test splits
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train)
    mars.fit(x_train2, y_train2)
    results.append(mars.summary())

term_ct_df = pd.DataFrame(x_final.columns.values)
term_ct_df['count'] = np.nan
for j in range(len(term_ct_df)):
    term = term_ct_df.iloc[j,0]
    count = 0
    for i in range(len(results)):
       count = Find_Pattern(results[i],term, count)
    term_ct_df['count'].loc[j] = count
term_ct_df['percent_present'] = term_ct_df['count']/201

#%%
'''
Model selection and optimization
'''

#%%
#train test split and autoregression preparation
#autoreg
oil2 =  auto_reg(oil2, 'oil_price', 7, "oil_price ~ month_no")
data = oil2.dropna()

data['refiner_margin'] = data['refiner_margin']
#train test split
train = data[data.month_no < 200]
test = data[data.month_no > 200]


#Most common MARS variables
xvar_all = ['retail_sales','indust_prod','us_cpi','glob_prod',
                      'us_indust_prod','new_rigs','oced_gdp',
                      'unem_rate','running_def','opec_prod','us_retail_sales',
                      'world_cpi','core_cpi']
#create combo list of all combinations of varaibles
all_combos = []
for c in range(len(xvar_all) + 1):
    all_combos += (list(itertools.combinations(xvar_all, c)))
    
#convert tuples to lists
xlist = []
for i in range(len(all_combos)):
    xlist.append(list(all_combos[i]))
#set up dataframe and lists to hold results
model = []
out_mse = []
#loop over all models and compute mse
for i in range(1,len(xlist)):
    mse = mse_2reg(train, test, xlist[i], 'oil_price', 'refiner_margin', 'gas_raw')
    model.append(xlist[i])
    out_mse.append(mse)
    print(i)
results = pd.DataFrame(out_mse, columns=['mse'])
results['model'] = model

#take the best 1000, add autoregression to smooth/improve
results = results.sort_values(by = 'mse')
top_results = results.head(1000)
#add auto to variables
for i in range(len(top_results)):
    top_results['model'].iloc[i].append('auto')
#run mse calc again with auto included
mse_auto = []
mse_oil = []
for i in range(len(top_results)):    
    mse_auto.append( mse_2reg(
            train, test, top_results['model'].iloc[i], 'oil_price', 'refiner_margin', 'gas_raw')) 
    mse_oil.append( mse_reg(train, top_results['model'].iloc[i], test)) 
top_results['mse_auto'] = mse_auto
top_results['mse_oil'] = mse_oil

#rescale MSE with minmax scaler
top_results['mse_auto_scale'] = minmax_scale(top_results,'mse_auto')
top_results['mse_oil_scale'] = minmax_scale(top_results,'mse_oil')

#add rescaled together
top_results['overall_mse'] = top_results['mse_auto_scale'] + top_results['mse_oil_scale']



#%%
'''
Best Model Pipeline: Oil price -> Gas Price
'''
#%%
#establish model

#model formula
mod_form_original = "oil_price ~ running_def + us_indust_prod + indust_prod + world_cpi + core_cpi + auto"
mod_form_best = "oil_price ~ running_def + us_indust_prod + indust_prod + us_retail_sales + world_cpi + core_cpi + auto"

mod_backmodel = "oil_price ~ us_cpi + us_indust_prod + core_cpi + indust_prod + retail_sales + world_cpi "


mod_form = mod_form_best
reg_auto = sm.formula.ols(formula = mod_form, data = train).fit()
reg_auto.summary()
#second model (convert oil to gas)
gas_reg = sm_reg(train, xvars=['oil_price','refiner_margin'], yvar = ['gas_raw'])
gas_reg.summary()

#predict oil over whole dataframe
whole_prediction = reg_auto.get_prediction(data)
whole_pred = whole_prediction.predicted_mean
#save in whole data
data['oil_pred'] = whole_pred

#predict gas from oil predictions over whole data frame
gas_preddata = data[['oil_pred', 'refiner_margin']]

gas_prediction = gas_reg.get_prediction(sm.add_constant(gas_preddata))
gas_pred = gas_prediction.predicted_mean

mse_total = np.mean((gas_pred - data.gas_raw)**2)
mse_outsample = np.mean((gas_pred.tolist()[-46:] - data.gas_raw[data.month_no > 200])**2)
mse_oil_outsample = np.mean((whole_pred.tolist()[-46:] - data.oil_price[data.month_no > 200])**2)

mae_outsample = np.mean(abs(gas_pred.tolist()[-46:] - data.gas_raw[data.month_no > 200]))
mae_oil_outsample = np.mean(abs(whole_pred.tolist()[-46:] - data.oil_price[data.month_no > 200]))

total_error = sum(abs(gas_pred.tolist() - data.gas_raw[data.month_no < 247]))
total_oil_error = sum(abs(whole_pred.tolist() - data.oil_price[data.month_no < 247]))

mae_total = np.mean(abs(gas_pred.tolist() - data.gas_raw[data.month_no < 247]))
mae_oil_total = np.mean(abs(whole_pred.tolist() - data.oil_price[data.month_no < 247]))

gas_total = sum(data.gas_raw[data.month_no < 247])
oil_total = sum(data.oil_price[data.month_no < 247])

#gas
plt.plot(data.month_no, data.gas_raw, color = 'blue')
plt.plot(data.month_no, gas_pred, color = 'red')
plt.title('Gas Price Prediction Model')
plt.ylabel('Gas Price')
plt.xlabel('Month of Dataset')
plt.text(150, 0.2, "**Trained on data before month 200", ha='left')
plt.text(40, 0.26, "*Out Sample MSE: 0.015", ha='left')
plt.text(40, 0.06, "*Out Sample MAE: 0.102", ha='left')
plt.text(150, 0.06, "*Total MAE: 0.107", ha='left')

#oil price
plt.plot(data.month_no, data.oil_price, color = 'blue')
plt.plot(data.month_no, whole_pred, color = 'red')
plt.title('Oil Price Prediction Model')
plt.ylabel('Oil Price')
plt.xlabel('Month of Dataset')
plt.text(150, -15, "**Trained on data before month 200", ha='left')
plt.text(40, -15, "*Out Sample MSE: 42.54", ha='left')
plt.text(40, -22, "*Out Sample MAE: 5.27", ha='left')
plt.text(150, -22, "*Total MAE: 4.63", ha='left')



#%%
'''
New Rig Analysis
'''
# %%
#timeseries US prod to rigs
fig, ax1 = plt.subplots()
ax1.set_xlabel('month number')
ax1.set_ylabel('New rigs - Blue', color='Blue')
ax1.plot(oil2.month_no, oil2.new_rigs)
ax2 = ax1.twinx()
ax2.set_ylabel('US Oil Production - Red', color='Red')
ax2.plot(oil2.us_prod, color='Red')
plt.show()

#scatter us prod to rigs
plt.scatter(oil2.new_rigs, oil2.us_prod, c=oil2.republican)
plt.ylabel('US Production')
plt.xlabel('New Oil Rigs (per month)')
plt.text(5200, 4.3e8, "Republicans Yellow, Democrats Purple", ha='center')

#%%

'''
Public acherage and new permits
Active acharage is INVERSELY related to oil production
New Permits are not related ro production

Permit issues came down under Obama, crashed at 2008 then rose under Trump
Biden has issued more permits in 2021 than Trump did in any year yet

Issued permits decently predicts number of active achers until 2018
Maybe sitting on permits? Also not all permits produce? Likely other explainations too
The correlation is loose too, so perhaps what drives permit seeking and useage are correlated
rather than the permit to the land?

'''
#%%
# Does new permits or public acherage predict US oil production?
plt.scatter(oil_yr.app_drill_perms, oil_yr.us_yr_prod)
plt.ylabel('US Production')
plt.xlabel('Approved Drilling Permits (Per Year)')
plt.title('US production vs Approved Drilling Permits')

plt.scatter(oil_yr.act_acherage, oil_yr.us_yr_prod)
plt.ylabel('US Production')
plt.xlabel('Active Drilling on Public Lands')
plt.title('US production vs Active Drilling Acreage')

#time series
fig, ax1 = plt.subplots()
ax1.set_xlabel('year number (of dataset not actual)')
ax1.set_ylabel('Permits Issued - Blue', color='Blue')
ax1.plot(oil_yr.year_num, oil_yr.app_drill_perms)
ax2 = ax1.twinx()
ax2.set_ylabel('Number of public actively producing achers - Red', color='Red')
ax2.plot(oil_yr.act_acherage, color='Red')
plt.axvline(12, color='purple')
plt.axvline(19, color='red')
plt.axvline(24, color='blue')
plt.text(15, 5.0e7, "Purple: Jan 2012, Red: Jan 2017, Blue: Jan 2021", ha='center')
plt.show()

#approved perms vs oil price
plt.scatter(oil_yr.oil_price, oil_yr.app_drill_perms)
plt.ylabel('Approved Permits')
plt.xlabel('Oil Price')

#%%
'''
Refiner Margins:
Record highs since june 2021


#changes in the margin correlate to changes in gas price
~1 cent change in margin = ~0.9 cent change in gas
R^2 = 0.43
'''
#%%
#record high refiner margins since mid 2021
plt.plot(oil2.month_no, oil2.refiner_margin)
plt.ylabel('Refiner Margin ($)')
plt.xlabel('Month Number')
#no correlation between gas price and margins until 2021 june
plt.scatter(oil2.refiner_margin,oil2.gas_raw)
plt.ylabel('Gas Price ($)')
plt.xlabel('Refiner Margin ($)')


#change in margin to price and change in price
oil2['margin_dif'] = oil2.refiner_margin.diff()
oil2['gas_dif'] = oil2.gas_raw.diff()

plt.scatter(oil2.margin_dif, oil2.gas_dif)
plt.ylabel('Delta Gas Price')
plt.xlabel('Delta Refiner Margin ($)')
plt.title('Change in Gas Prices vs Change in Refiner Margins')
plt.text(0.3, -0.57, "R^2 = 0.36", ha='center')


margin_changereg = smf.ols(formula="gas_dif ~ margin_dif",
                      data=oil2).fit(cov='HC3')
margin_changereg.summary()

margin_changereg = smf.ols(formula="gas_raw ~ refiner_margin",
                      data=oil2).fit(cov='HC3')
margin_changereg.summary()

#%%

'''
What Drives Production?

US production is not corrlated to oil price, gas price, pesidential party and others
US production is correlated to retail sales and US CPI and others

Global production is not correlated to US consumption, presidential party or unemployment rate and others
Global production is correlated to oil/gas price, global and US CPI, indust production, retail sales, running def, consumption and others
'''
#%%
#get correlations
correlates = oil2.corr()
globprod_corr = correlates['glob_prod']
usprod_corr = correlates['us_prod']
#Mark only those which coeff > 0.5
globprod_corr_ind = abs(globprod_corr) > 0.5
usprod_corr_ind = abs(usprod_corr) > 0.5

both = {
        "Global Production" :globprod_corr,
        "US Production" : usprod_corr,
        "Global Production Ind" :globprod_corr_ind,
        "US Production Ind" : usprod_corr_ind
        }

both_corrs = pd.concat(both, axis = 1)

plt.scatter(oil2.glob_con, oil2.glob_prod)
plt.title('Global Production vs Global Consumption')
plt.ylabel('Global Consumption')
plt.xlabel('Global Production')
plt.text(2.8e9, 2.4e9, "R^2 = 0.95", ha='center')

plt.scatter(oil2.indust_prod, oil2.glob_prod)
plt.title('Global Production vs Global Industrial Production')
plt.ylabel('Global Industrial Production')
plt.xlabel('Global Production')
plt.text(1.5e12, 2.4e9, "R^2 = 0.97", ha='center')

plt.scatter(oil2.world_cpi, oil2.glob_prod)
plt.ylabel('Global CPI')
plt.title('Global Production vs Global CPI')
plt.xlabel('Global Production')
plt.text(110, 2.4e9, "R^2 = 0.96", ha='center')

plt.scatter(oil2.oil_price, oil2.glob_prod)
plt.title('Oil Price to Production')
plt.xlabel('Oil Price')
plt.ylabel('Global Production')
plt.text(110, 2.4e9, "R^2 = 0.27", ha='center')

oil2['post_2011'] = np.where(oil2.year > 2011, 1, 0)


plt.scatter(oil2.us_prod, oil2.glob_prod, c = oil2.post_2011)
plt.title('US Production to Global Production')
plt.xlabel('US Production')
plt.ylabel('Global Production')
plt.text(3.3e8, 2.4e9, "R^2 = 0.60", ha='center')

plt.scatter(oil2.us_retail_sales, oil2.us_prod, c = oil2.post_2011)
plt.title('US Production vs US Retail Sales')
plt.xlabel('US Retail Sales')
plt.ylabel('US Production')
plt.text(130, 2.2e8, "R^2 = 0.64", ha='center')

plt.scatter(oil2.gas_raw, oil2.us_prod, c = oil2.post_2011)
plt.title('US Production vs Gas Price')
plt.xlabel('Gas Price')
plt.ylabel('US Production')
plt.text(1.25, 3.5e8, "R^2 = 0.05", ha='center')


#what drives gas price

plt.scatter(oil2.oil_price, oil2.gas_raw)
plt.title('Oil Price vs Gas Price')
plt.xlabel('Oil Price')
plt.ylabel('Gas Price')
plt.text(30, 3.5, "R^2 = 0.93", ha='center')


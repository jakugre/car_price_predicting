### it takes around 200 second for the whole code to run 
#data obtained from https://www.kaggle.com/bartoszpieniak/poland-cars-for-sale-dataset?select=Car_sale_ads.csv

import pandas as pd 
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LinearRegression



data = pd.read_csv('Car_sale_ads.csv')
x,y = data.shape

print('Names of columns in our data file: ',data.columns)

#getting rid of insignificant data
data['Vehicle_brand_and_model'] = data['Vehicle_brand'] + ' ' +data['Vehicle_model']
data.drop(['CO2_emissions','First_registration_date','First_owner','Index','Offer_publication_date',
			'Offer_location','Vehicle_version','Currency','Features','Colour','Drive','Vehicle_brand',
			'Vehicle_model','Vehicle_generation'],inplace=True,axis=1)

print('Names of columns that are significant for our model: ',data.columns)

##### REPLACING TEXT DATA WITH NUMERIC REFERENCES - SEE car_references.xlsx FOR INFO #####
	
#replacing countries with numbers
unique_list = list(data.Origin_country.unique())
for country in unique_list:
	if country != 'Poland':
		data = data.replace({'Origin_country':{country:0}})
	else:
		data = data.replace({'Origin_country':{country:1}})

#replacing vehicle_brand_and_model with numbers
vehicle_dict = {}
vehicle_list = list(data.Vehicle_brand_and_model.unique())
for i in range(len(vehicle_list)):
	vehicle_dict[vehicle_list[i]] = i 
	data = data.replace({'Vehicle_brand_and_model':{vehicle_list[i]:i}})


#replacing fuel_types with numbers
fuel_list = list(data.Fuel_type.unique())
for i in range(len(fuel_list)):
	data = data.replace({'Fuel_type':{fuel_list[i]:i}})

#replacing type with numbers 
type_list = list(data.Type.unique())
for i in range(len(type_list)):
	data = data.replace({'Type':{type_list[i]:i}})

#replacing transmission and condition with numbers
data = data.replace({'Transmission':{'Manual':0,'Automatic':1,np.nan:2},'Condition':{'New':0,'Used':1}})

##### END #####



#search for columns with missing values
missing_val_count_by_column = (data.isnull().sum())
print('Number of missing values in columns (if any): ',missing_val_count_by_column[missing_val_count_by_column > 0])

#since number of missing values in columns is very small (less than 0.5% of all values), it is easier to drop any rows with missing values in them
data = data.dropna(axis=0,how='any',inplace=False)



#function to calculate average price of car by name and year - used to check how far the predicition is from mean value
def get_price(car_name,year):
	car_index = vehicle_dict.get(car_name)
	df = data.loc[data['Vehicle_brand_and_model'] == car_index]
	df = df.loc[df['Production_year'] == year]
	
	return df['Price'].mean()
	






train_features = data.copy()
train_labels = train_features.pop('Price')
X_train, X_valid, Y_train, Y_valid = train_test_split(train_features, train_labels, train_size=0.9, test_size=0.1, random_state=0)



### RANDOM FOREST REGRESSION ###
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, Y_train)
price_preds = forest_model.predict(X_valid)
#print(forest_model.predict([train_features.iloc[-1,:]]),train_labels.iloc[-1])
print('R2 score for predicted labels - Random Forest: ',forest_model.score(X_valid,Y_valid))


### EXTRA TREES REGRESSION ###
extra_tree_model = ExtraTreesRegressor(random_state=1)
extra_tree_model.fit(X_train, Y_train)
extra_price_preds = extra_tree_model.predict(X_valid)
print('R2 score for predicted labels - Extra Trees: ',extra_tree_model.score(X_valid,Y_valid))

### GRADIENT BOOST REGRESSION ###
grad_model = GradientBoostingRegressor(random_state=1)
grad_model.fit(X_train, Y_train)
grad_price_preds = grad_model.predict(X_valid)
print('R2 score for predicted labels - Gradient Boost: ',r2_score(np.array(Y_valid), grad_price_preds))

### XGB BOOST REGRESSION ###
XGB_model = XGBRegressor(random_state=1)
XGB_model.fit(X_train, Y_train)
XGB_price_preds = XGB_model.predict(X_valid)
print('R2 score for predicted labels - XGB Boost: ',r2_score(np.array(Y_valid), XGB_price_preds))

### LINEAR REGRESSION ###
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
linear_price_preds = linear_model.predict(X_valid)
print('R2 score for predicted labels - Linear Regression: ',r2_score(np.array(Y_valid), linear_price_preds))
import pandas as pd 
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score



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
vehicle_list = list(data.Vehicle_brand_and_model.unique())
for i in range(len(vehicle_list)):
	data = data.replace({'Vehicle_brand_and_model':{vehicle_list[i]:i+1}})

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



train_features = data.copy()
train_labels = train_features.pop('Price')
X_train, X_valid, Y_train, Y_valid = train_test_split(train_features, train_labels, train_size=0.8, test_size=0.2, random_state=0)



forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, Y_train)
price_preds = forest_model.predict(X_valid)
#print(forest_model.predict([train_features.iloc[-1,:]]),train_labels.iloc[-1])
print('R2 score for predicted labels: ',r2_score(np.array(Y_valid), price_preds))
####kissan analytics
import sweetviz 
import pandas as pd                    # Package for data manipulations
data = pd.read_csv("D://Kisan_analytics//final_data2.csv")     # importing the dataset
data.head(10)                          # Returns first 10 rows
data.shape                             # Returns no. of rows and columns
data.info()                            # Returns no. of columns, column labels, column data types, memory usage etc.
d= data.describe()                        # Returns description of the data 
data.isna().sum()                      # No null values
data.Soil_Type.unique()
data.head()
data.columns
data.Crop.unique()
###
###EDA####
# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
data.ROI.mean() 
data.ROI.median()
data.ROI.mode()
# Yield
data.Yield.mean() 
data.Yield.median()
data.Yield.mode()
#MSP
data.MSP_FRP_.mean() 
data.MSP_FRP_.median()
data.MSP_FRP_.mode()
#Total cost of production
data.Total_cost_of_production.mean() 
data.Total_cost_of_production.median()
data.Total_cost_of_production.mode()

# Measures of Dispersion / Second moment business decision
data.ROI.var() # variance
data.ROI.std() # standard deviation
range = max(data.ROI) - min(data.ROI) # range
range
data.Total_cost_of_production.var() # variance
data.Total_cost_of_production.std() # standard deviation
range = max(data.Total_cost_of_production) - min(data.Total_cost_of_production) # range
range


# Third moment business decision
data.ROI.skew()
data.Total_cost_of_production.skew()

# Fourth moment business decision
data.ROI.kurt()
data.Total_cost_of_production.skew()

####outliers treatment

# checking for the outliers
import seaborn as sns
sns.boxplot(data.ROI)
#outliers
sns.boxplot(data.Yield)
# outliers
sns.boxplot(data.MSP_FRP_)
# No  outliers
sns.boxplot(data.Total_cost_of_production)
# outliers
# Detection of outliers (find limits for ROI based on IQR)
IQR = data['ROI'].quantile(0.75) - data['ROI'].quantile(0.25)
lower_limit = data['ROI'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['ROI'].quantile(0.75) + (IQR * 1.5)

#Total cost of production
IQR = data['Total_cost_of_production'].quantile(0.75) - data['Total_cost_of_production'].quantile(0.25)
lower_limit = data['Total_cost_of_production'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Total_cost_of_production'].quantile(0.75) + (IQR * 1.5)

#Yield
IQR = data['Yield'].quantile(0.75) - data['Yield'].quantile(0.25)
lower_limit = data['Yield'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Yield'].quantile(0.75) + (IQR * 1.5)
###Data visualisation###
### Districts having highest yield year wise ###
y1 = data[data.Year == 2013].groupby("District_")["Yield"].sum().sort_values(ascending = False)
''' Nizamabad has highest yield followed by Medak and Khammam in 2013'''
y2 = data[data.Year == 2014].groupby("District_")["Yield"].sum().sort_values(ascending = False)
''' Nizamabad has highest yield followed by Karimnagar and Hyderabad in 2014'''
y3 = data[data.Year == 2015].groupby("District_")["Yield"].sum().sort_values(ascending = False)
''' Adilabad has highest yield followed by Karimnagar and Hyderabad in 2015'''
y4 = data[data.Year == 2016].groupby("District_")["Yield"].sum().sort_values(ascending = False)
''' Nizamabad has highest yield followed by Adilabad and Medak in 2016'''
y5 = data[data.Year == 2017].groupby("District_")["Yield"].sum().sort_values(ascending = False)
''' Nizamabad has highest yield followed by Warangal and Adilabad in 2017'''

# Total yield of different crops season wise
crop_season = data.groupby(['Season', 'Crop'])["Yield"].sum()
cys = pd.DataFrame({"Yield_": crop_season}).reset_index()

crop_rabi = cys[cys['Season'] == 'Rabi']
crop_kharif = cys[cys['Season'] == 'Kharif']

""" Sugarcane, maize, rice, groundnuts grown in kharif season has highest yield followed by cheakpea,sorghum 
and wheat grown in rabi season """

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data.Crop, data.Yield)
data.groupby("Crop")["Yield"].mean().plot.bar()
plt.show()
data.groupby("Crop")["Total_cost_of_production"].mean().plot.bar()
####correlation####
corrl = data.corr()

crop_wise_roi = data.groupby("Crop")["ROI"].mean().plot.bar()
plt.show()

district_crop_yield = data.groupby(['District_'])
index = list(district_crop_yield.indices.keys())
district_crop_yield.sum()[["Total_cost_of_production", "Yield" ]].plot(kind='bar',figsize=(12,7))

pd.pivot_table(data, index = "Crop", columns = "District_", values = "ROI")
# profit on different crops in different district
yield_year_wise = pd.pivot_table(data, index = "Crop", columns = "Year", values = "Yield")
# Yield of different crops year wise
# HISTOGRAM
plt.hist(data['ROI'], color= "red")
plt.hist(data['Yield'], color= "blue")
plt.hist(data['Total_cost_of_production'], color= "orange")

District_ = data['District_'].unique()
District_

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['District_'] = le.fit_transform(data['District_'])

int_encoded_district= le.fit_transform(District_)
int_encoded_district
district_dict = {District_ : label for District_ , label in zip(District_, int_encoded_district)}

Season = data.Season.unique()
int_encoded_season= le.fit_transform(['Season'])
int_encoded_season
Season_dict = {Season: label for Season , label in zip(Season, int_encoded_season)}

Crop = data.Crop.unique()
int_encoded_crop= le.fit_transform(['Crop'])
int_encoded_crop
Crop_dict = {Crop: label for Crop , label in zip(Crop, int_encoded_crop)}

Soil_Type = data.Soil_Type.unique()
int_encoded_Soil_Type= le.fit_transform(['Soil_Type'])
int_encoded_Soil_Type
Crop_dict = {Soil_Type : label for Soil_Type , label in zip(Soil_Type, int_encoded_Soil_Type)}

crop_name = data['Crop'].unique()
crop_name

int_encoded_crop= le.fit_transform(crop_name)
int_encoded_crop

crop_name_dict={label:crop_name for crop_name , label in zip( crop_name , int_encoded_crop )}
crop_yield_dict={crop_name:label for crop_name , label in zip( crop_name , int_encoded_crop )}

crop_name_dict
crop_yield_dict

data['District_'] = le.fit_transform(data['District_'])
data['Category_of_Crop'] = le.fit_transform(data['Category_of_Crop'])
data['Season'] = le.fit_transform(data['Season'])
data['Soil_Type'] = le.fit_transform(data['Soil_Type'])
data["Year"] = le.fit_transform(data["Year"])
labels = data['Crop']


X = data[['District_','Year', 'Season', 'Optimum_pH','N', 'P', 'K', 'Soil_Type','Yield', 'Avg_rainfall', 'Avg_Temp']]
#X = data[['District_','Year','Season','Optimum_pH', 'N', 'P', 'K','Soil_Type','Yield' ,'Avg_rainfall', 'Avg_Temp','MSP_FRP_']]
Y = data['Crop']

# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size = 0.2,random_state = 0)
Xtrain.shape
Xtest.shape
my_report = sweetviz.analyze([Xtrain, "Train"],target_feat='Yield')
### create a whole report in form of HTML file
my_report.show_html('Report.html')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain_sc = sc.fit_transform(Xtrain)
Xtest_sc = sc.fit_transform(Xtest)
from sklearn.linear_model import LogisticRegression

# Create one-vs-rest logistic regression object
lgr = LogisticRegression(penalty='l2',C = 1.0,fit_intercept=True, solver="lbfgs",class_weight=None, random_state=None, max_iter=100, multi_class='ovr')
# Train model
model = lgr.fit(Xtrain_sc, Ytrain)

predicted_values = model.predict(Xtest)
model_classes= model.classes_

import numpy as np
data1 = np.array([[2,4,1,7,20, 60,20,8,167.98,110, 22.07]])   
prd1 = model.predict(data1)
prd1

# View predicted probabilities

'''In one-vs-rest logistic regression (OVR) a separate model is trained for each class predicted whether an observation
 is that class or not (thus making it a binary classification problem). It assumes that each classification problem 
 (e.g. class 0 or not) is independent.'''

crop_prob = model.predict_proba(data1)
predicted_values = crop_prob.tolist()

print(np.flip(np.argsort(predicted_values)))
sorted_label= np.flip(np.argsort(predicted_values))[0]
print(sorted_label)
list_crops = sorted_label.tolist()
list_crops
'''
def recommended_crops(top = 3):
  return [crop_name_dict.get(sorted_label[i]) for i in range(top)]

print(recommended_crops(3))
'''
z = ['Chickpeas','Cotton','Finger Millet','Groundnuts','Maize','Mustard','Pearl Millet','Pigeonpea','Rice',	'Sesamum','Sorghum','Sugarcane','Sunflower','Wheat']

print(crop_prob)
for r in crop_prob:
    sorted_dict = sorted({z1: r1 for r1, z1 in zip(r, z)}.items(), key=lambda x: x[1],reverse = True)
    print([c for c, _ in sorted_dict][:3])

#saving the model
import pickle
with open('log_pickle1.pkl','wb') as f:
    pickle.dump(model,f)
with open('log_pickle1.pkl','rb') as f:
    log_pickle1 = pickle.load(f)


with open('district_dict_pickle.pkl','wb') as f:
    pickle.dump(district_dict,f)
with open('district_dict_pickle.pkl','rb') as f:
    district_dict_pickle = pickle.load(f)


with open('Season_dict_pickle.pkl','wb') as f:
    pickle.dump(Season_dict,f)
with open('Season_dict_pickle.pkl','rb') as f:
    Season_dict_pickle = pickle.load(f)


print(Crop_dict)
with open('Crop_dict_pickle.pkl','wb') as f:
    pickle.dump(Crop_dict,f)
with open('Crop_dict_pickle.pkl','rb') as f:
    Crop_dict_pickle = pickle.load(f)

with open('crop_name_dict_pickle.pkl','wb') as f:
    pickle.dump(crop_name_dict,f)
with open('crop_name_dict_pickle.pkl','rb') as f:
    crop_name_dict_pickle = pickle.load(f)


with open('crop_yield_dict_pickle.pkl','wb') as f:
    pickle.dump(crop_yield_dict,f)
with open('crop_yield_dict_pickle.pkl','rb') as f:
    crop_yield_dict_pickle = pickle.load(f)





    
'''
with open('Optimum_pH_dict_pickle.pkl','wb') as f:
    pickle.dump(Optimum_pH_dict,f)
with open('Optimum_pH_dict_pickle.pkl','rb') as f:
    Optimum_pH_dict_pickle = pickle.load(f)


with open('N_dict_pickle.pkl','wb') as f:
    pickle.dump(N_dict,f)
with open('N_dict_pickle.pkl','rb') as f:
    N_dict_pickle = pickle.load(f)

with open('P_dict_pickle.pkl','wb') as f:
    pickle.dump(P_dict,f)
with open('P_dict_pickle.pkl','rb') as f:
    P_dict_pickle = pickle.load(f)


with open('K_dict_pickle.pkl','wb') as f:
    pickle.dump(K_dict,f)
with open('K_dict_pickle.pkl','rb') as f:
    K_dict_pickle = pickle.load(f)


with open('Soil_Type_dict_pickle.pkl','wb') as f:
    pickle.dump(Soil_Type_dict,f)
with open('Soil_Type_dict_pickle.pkl','rb') as f:
    Soil_Type_dict_pickle = pickle.load(f)


with open('Avg_rainfall_dict_pickle.pkl','wb') as f:
    pickle.dump(Avg_rainfall_dict,f)
with open('Avg_rainfall_dict_pickle.pkl','rb') as f:
    Avg_rainfall_dict_pickle = pickle.load(f)


with open('Avg_Temp_dict_pickle.pkl','wb') as f:
    pickle.dump(Avg_Temp_dict,f)
with open('Avg_Temp_dict_pickle.pkl','rb') as f:
    Avg_Temp_dict_pickle = pickle.load(f)
'''









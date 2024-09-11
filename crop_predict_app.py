#import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
loaded_model = pickle.load(open('log_pickle1.pkl','rb'))
#loaded_model2 = pickle.load(open('yield_pkl1.pkl','rb'))
# loaded_model3 = pickle.load(open('roi_pkl.pkl','rb'))
district_dict = pickle.load(open('district_dict_pickle.pkl','rb'))
Season_dict =  pickle.load(open('Season_dict_pickle.pkl','rb'))
Crop_dict = pickle.load(open('Crop_dict_pickle.pkl','rb'))
Soil_Type_dict = pickle.load(open('Soil_Type_dict_pickle.pkl','rb'))
crop_name_dict = pickle.load(open('crop_name_dict_pickle.pkl','rb'))
crop_yield_dict  =  pickle.load(open('crop_name_dict_pickle.pkl','rb'))
# creating function for prediction


def crop_predict(District_,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp):
    District_ = district_dict.get(District_[0])
    Season = Season_dict.get(Season[0])
    Soil_Type = Soil_Type_dict.get(Soil_Type[0])
    
    prd1 = loaded_model.predict([[District_,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp]])
    
    result =loaded_model.predict_proba([[District_,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp]])
    print(result)
    r2 =result.tolist()
    z = ['Chickpeas','Cotton','Finger Millet','Groundnuts','Maize','Mustard','Pearl Millet','Pigeonpea','Rice',	'Sesamum','Sorghum','Sugarcane','Sunflower','Wheat']
    print(result)
    
    for r in result:
        sorted_dict = sorted({z1: r1 for r1, z1 in zip(r, z)}.items(), key=lambda x: x[1],reverse = True)
        return ([c for c, _ in sorted_dict][:3])

def main():
    st.title('Crop Prediction WebApp')
    District_= st.selectbox("Enter District", list(district_dict.items()), 0 , format_func=lambda o: o[0])
    Season = st.selectbox("Season", list(Season_dict.items()), 0 , format_func=lambda o: o[0])
    Optimum_pH = st.number_input('Optimum pH')
    N= st.number_input('Nitrogen in kg per hectare')  
    P= st.number_input('Phosphorus in kg per hectare') 
    K= st.number_input('Potassium in kg per hectare') 
    Soil_Type = st.selectbox("Enter the type of soil" ,list(Soil_Type_dict.items()), 0 , format_func=lambda o: o[0])
    Avg_rainfall= st.number_input('Average rainfall in mm')  
    Avg_Temp= st.number_input('Average temperature in degree celsius')    
    Crops = " "
    #data1 = np.array([District_,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp])
    if st.button("Crop"): 
        Crops = crop_predict(District_,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp)
#        yield_pred = yield_predict(District_,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp,Crops)
        st.success(Crops)
    
if __name__ == '__main__':
	main()
                 
        
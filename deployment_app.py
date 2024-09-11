import numpy as np     
import pickle
import streamlit as st

loaded_model = pickle.load(open('log_pickle1.pkl','rb'))
district_dict = pickle.load(open('district_dict_pickle.pkl','rb')) 
Season_dict = pickle.load(open('Season_dict_pickle.pkl','rb')) 
Crop_dict = pickle.load(open('Crop_dict_pickle.pkl','rb')) 
Optimum_pH_dict = pickle.load(open('Optimum_pH_dict_pickle.pkl','rb')) 
N_dict = pickle.load(open('N_dict_pickle.pkl','rb')) 
P_dict = pickle.load(open('P_dict_pickle.pkl','rb')) 
K_dict = pickle.load(open('K_dict_pickle.pkl','rb')) 
Soil_Type_dict = pickle.load(open('Soil_Type_dict_pickle.pkl','rb')) 
Avg_rainfall_dict= pickle.load(open('Avg_rainfall_dict_pickle.pkl','rb')) 
Avg_Temp_dict = pickle.load(open('Avg_Temp_dict_pickle.pkl','rb')) 

# creating function for prediction
def crop_predict(District_,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp):
    District_ = district_dict.get(District_[0])
    Season= Season_dict.get(Season[0])
    Optimum_pH = Optimum_pH_dict.get(Optimum_pH[0])
    N = N_dict.get(N[0])
    P = P_dict.get(P[0])
    K = K_dict.get(K[0])
    Soil_Type = Soil_Type_dict.get(Soil_Type[0])
    Avg_rainfall = Avg_rainfall_dict.get(Avg_rainfall[0])
    Avg_Temp = Avg_Temp_dict.get(Avg_Temp[0])
    predicted_values = crop_prob.tolist()
    sorted_label = np.flip(np.argsort(predicted_values))[0]
    return [Crop_dict.get(sorted_label[i]) for i in range(3)]

def main():
    st.title("Crop Prediction Web App")
    # Getting input data from user
    District_1= st.selectbox('Enter district' ,('Hyderabad', 'Nizamabad', 'Medak', 'Mahabubnagar', 'Nalgonda',
           'Warangal', 'Khammam', 'Karimnagar', 'Adilabad'))
    st.write('Selected district:' , District_1)
    Season= st.selectbox('Enter season',('Rabi','Kharif'))
    st.write('Selected season:' , Season)
    Optimum_pH = st.text_input('Optimum pH')
    N= st.text_input('Nitrogen')  
    st.write("unit : Kg per hectare")
    P= st.text_input('Phosphorus') 
    st.write("unit : kg per hectare")
    K= st.text_input('Potassium') 
    st.write("unit : kg per hectare")
    Soil_Type = st.selectbox('Enter type of soil',('Sandy loam', 'Black soil', 'Red laterite ',
           'Colluvial soil', 'Alluvial loam', 'Clay loam','Black cotton soil'))
    st.write('Selected soil type:', Soil_Type)
    Avg_rainfall= st.text_input('Average rainfall')  
    st.write("unit: mm")
    Avg_Temp= st.text_input('Average temperature')    
    st.write("unit: degree celsius")
    Crops = " "
    data1 = np.array([District_1,Season, Optimum_pH, N, P, K, Soil_Type, Avg_rainfall, Avg_Temp])
    if st.button("Crop"): 
        Crops = crop_predict(data1)
        #Crops = crop_predict(1,3,1,7,20, 60,20,8,167.98,1100, 22.07)
    st.success(Crops)
    st.write("Best three crops to be grown")
   

if __name__ == '__main__':
	main()
                 
        
# agriculture_analytics

The given model considers nine features out of which three features are categorical and hence label encoded.
Categorical features were extracted separately and converted to dictionary form.
Logistic regression model was applied on the dataset which gives all the probabilities of the individual crops and then we have   taken all the probabilities as array and then sorted and given top 3 crops with highest probabilities as top three crops suitable for that particular district.
In one-vs-rest logistic regression (OVR) a separate model is trained  for each class predicted whether an observation is that class or not (thus making it a binary classification problem). 
It assumes that each classification problem (e.g. class 0 or not) is independent.
After the prediction, model and categorical data in dictionary form were picked.![image](https://github.com/user-attachments/assets/33aee63b-eb17-4311-b509-0a1a91455238)

Input variables used
In order to identify the top 3 suitable crops for the farmers from the raw data, 

District : Different districts of Telangana were taken as one of the input
Season : The season in which crop is grown( Kharif , Rabi)
Crop :All the crops grown in the different districts of Telangana were taken as one of the input
Yield  :yield of the particular crop was collected from 2011 to 2017(in kg per hectares).
Rainfall : Average rainfall of the particular district was recorded(in millimeters).
Average Temperature : The average temperature of the district recorded as taken as input(in centigrade).
N : The amount of nitrogen based fertilizer needed for the crop to grow (in kg per hectare)
P : The amount of Phosphorus based fertilizer needed for the crop to grow (in kg per hectare)
K : The amount of Potassium based fertilizer needed for the crop to grow (in kg per hectare)
Soil Type : Type of the soil in which the crop was grown.
pH Value : The pH value of the soil suitable for the growth of the crop.
![image](https://github.com/user-attachments/assets/c9622faa-2610-4fca-8000-16fd3b3837ef)

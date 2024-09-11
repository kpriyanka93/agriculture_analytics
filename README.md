# agriculture_analytics

The given model considers nine features out of which three features are categorical and hence label encoded.
Categorical features were extracted separately and converted to dictionary form.
Logistic regression model was applied on the dataset which gives all the probabilities of the individual crops and then we have   taken all the probabilities as array and then sorted and given top 3 crops with highest probabilities as top three crops suitable for that particular district.
In one-vs-rest logistic regression (OVR) a separate model is trained  for each class predicted whether an observation is that class or not (thus making it a binary classification problem). 
It assumes that each classification problem (e.g. class 0 or not) is independent.
After the prediction, model and categorical data in dictionary form were picked.![image](https://github.com/user-attachments/assets/33aee63b-eb17-4311-b509-0a1a91455238)


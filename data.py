import pandas as pd   
import matplotlib_inline 
import matplotlib.pyplot as plt 
import numpy as np     
from pandas.plotting import scatter_matrix
df = pd.read_csv('housing.csv')  
#print(df.head())                                          #to view top 5 rows in dataset. 
#print(df.info())                                          #to get a quick description of the data like the total number of rows, each attribute’s type and the number of nonnull values. 
#print(df["ocean_proximity"].value_counts())               #to find no of categories and no of districts belonging to each category. 
#print(df.describe())                                      #to get a summary of the numerical attributes. 
#df.hist(bins = 70, figsize = (40,50))                     #bins=spacing between bars of histogram, figsize=(width,height)
#plt.show() 
df_numeric=df.select_dtypes(include=[np.number])           #drops non numeric columns
corr_matrix=df_numeric.corr()                              #a matrix to find correlation
#print(corr_matrix["median_house_value"].sort_values(ascending=False))           #finds correlation b/w median house value and other attributes
#attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]        #another method to find correlation
#scatter_matrix(df[attributes], figsize=(20,16)) 
#plt.show() 
#df.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.8)                     
#plt.show()  
df["rooms_per_household"]=df["total_rooms"]/df["households"]                         #creates now category named rooms per household 
df["bedrooms_per_room"]=df["total_bedrooms"]/df["total_rooms"]                       #creates new category named bedrooms per room
df["population_per_household"]=df["population"]/df["households"]                     #creates new category named population per household
corr_matrix=df_numeric.corr()  
print(corr_matrix["median_house_value"].sort_values(ascending=False)) 

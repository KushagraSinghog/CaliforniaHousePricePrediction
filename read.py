import pandas as pd   
import matplotlib.pyplot as plt 
import numpy as np     
df = pd.read_csv('housing.csv') 

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)   #splitting into train and test set while keeping the records of the sets same
#print(len(train_set)) 
#print(len(test_set))     
df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])   
#pd.cut is used to set data values into bins. df["median_income"] column is the input data for categorization.      
#bins=[0., 1.5, 3.0, 4.5, 6., np.inf] specifies the bin edges. Each bin includes the left edge but excludes the right edge, except for the last bin which includes both edges.    
#labels=[1, 2, 3, 4, 5] assigns these labels to the bins.
#df["income_cat"].hist() 
#plt.show()

from sklearn.model_selection import StratifiedShuffleSplit                   #used for making stratified test and training set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)   #split is the object for stratified splitting
for train_index, test_index in split.split(df, df["income_cat"]):            #loop for splitting the data into stratified training set and stratified test set  
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index] 
#print(strat_test_set["income_cat"].value_counts())  
strat_train_set = strat_train_set.drop("income_cat", axis=1)                 #used to drop the income cat
strat_test_set = strat_test_set.drop("income_cat", axis=1) 
df=strat_train_set.copy()                                                    #used to copy the training set
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=df["population"]/100, label="population", c=df["median_house_value"], cmap=plt.get_cmap("jet"), colorbar=True)              #used to create a scatterplot showing its population and median house value
plt.legend() 
#plt.show()  

df_numeric=df.select_dtypes(include=[np.number])                             #drops non numeric columns
corr_matrix=df_numeric.corr()                                                #a matrix to find correlation
#print(corr_matrix["median_house_value"].sort_values(ascending=False))           #finds correlation b/w median house value and other attributes

df.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.8) 
#plt.show()  
df=strat_train_set.drop("median_house_value", axis=1)                           #removes the median_house_value column from strat_train_set, leaving only the feature variables. Now df conatins all features except median house value
df_labels=strat_train_set["median_house_value"].copy()                          #creates a separate variable (df_labels) containing only the target variable (median_house_value).   

from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(strategy="median")                                     #creating the simpleimputer instance
df_num=df.drop("ocean_proximity", axis=1) 
imputer.fit(df_num)                                                          #fills the missing values with median values in the entire dataset
#print(imputer.statistics_)   
x=imputer.transform(df_num)                                                  #using the trained imputer to transform the dataset by filling the missing values
df_tr=pd.DataFrame(x, columns=df_num.columns, index=df_num.index)            #transforms the array back to dataframe form 
df_cat=df[["ocean_proximity"]] 
#print(df_cat.head(10))  

from sklearn.preprocessing import OrdinalEncoder 
og_en=OrdinalEncoder()                                                       #encoder object
df_cat_en=og_en.fit_transform(df_cat)                                        #Fits the encoder to the ocean_proximity values and transforms them into integers based on lexicographical order.
#print(df_cat_en[:10]) 
#print(og_en.categories_)   

from sklearn.preprocessing import OneHotEncoder 
cat_en= OneHotEncoder()                                                      #onehotencoder object 
df_cat_1hot= cat_en.fit_transform(df_cat)                                    #fits the onehotencoder to df_cat and transforms it
#print(df_cat_1hot.toarray())                                                 #gives the output as dense numpy array instead of scipy sparse matrix 

from sklearn.base import BaseEstimator , TransformerMixin  
rooms_ix , bedrooms_ix, pop_ix, hhlds_ix = 3,4,5,6                           #indices of columns
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):              #importing base estimator and transformermixing class
    def __init__(self, add_bedrooms_per_room= True):                         #costructor to add bedrooms per room and decides whether to calculate add bedrooms per room feature or not
        self.add_bedrooms_per_room = add_bedrooms_per_room 
    def fit(self, X, y=None):                                                #doesnt do anything but is needed to comply with scikit learn api
        return self 
    def transform(self, X):                                                  #function to take input x and create new features
        rooms_per_hhld = X[:, rooms_ix] / X[:, hhlds_ix] 
        pop_per_hhld = X[:, pop_ix] / X[:, hhlds_ix] 
        if self.add_bedrooms_per_room:                                               #if add bedrooms per room feature is true, then calculate bedrooms per room
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix] 
            return np.c_[X, rooms_per_hhld, pop_per_hhld, bedrooms_per_room] 
        else: 
            return np.c_[X, rooms_per_hhld, pop_per_hhld] 
        

from sklearn.pipeline import Pipeline                                        
from sklearn.preprocessing import StandardScaler 
num_pipeline = Pipeline( [ ('imputer', SimpleImputer(strategy="median")), ('att_adder', CombinedAttributesAdder()), ('std_scaler', StandardScaler()) ]) 
df_num_tr=num_pipeline.fit_transform(df_num)                                 #transformer pipeline created 

from sklearn.compose import ColumnTransformer 
num_att=list(df_num) 
cat_att=["ocean_proximity"] 
full_pipeline=ColumnTransformer( [ ("num", num_pipeline, num_att), ("cat", OneHotEncoder(), cat_att) ] ) 
df_prepared=full_pipeline.fit_transform(df)                                  #applying the pipeline to complete data    
 
from xgboost import XGBRegressor                                                      #using xgbregressor
xbg_reg=XGBRegressor(random_state=42) 

from sklearn.model_selection import GridSearchCV                                                    #to search for best parameter combinations for the model
param_grid=[ { 'n_estimators': [50,100,150], 'max_depth': [3,5,7], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.9, 1.0], 'colsample_bytree': [0.7, 0.9, 1.0] } ]  
grid_search=GridSearchCV(xbg_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)                #verbose controls how much information gets printed while the model is training, n_jobs controls how any cpu cores will be used
grid_search.fit(df_prepared, df_labels) 
#print(grid_search.best_estimator_) 
#print(np.sqrt(-grid_search.best_score_))  

final_model = grid_search.best_estimator_                                                      #it saves the model with best hyperparameters

from sklearn.metrics import mean_squared_error, r2_score                                       #training the model
xgb_pred=final_model.predict(df_prepared) 
xgb_mse=mean_squared_error(df_labels, xgb_pred) 
xgb_rmse=np.sqrt(xgb_mse) 
xgb_r2=r2_score(df_labels, xgb_pred)

#print(xgb_rmse)   
#print(xgb_r2)   


X_test=strat_test_set.drop("median_house_value", axis=1) 
y_test=strat_test_set["median_house_value"].copy() 

X_test_prepared=full_pipeline.transform(X_test)                                                #test set prepared
final_pred=final_model.predict(X_test_prepared) 

final_mse=mean_squared_error(y_test, final_pred)                                             #checking accuracy on test set
final_rmse=np.sqrt(final_mse) 
final_r2=r2_score(y_test, final_pred)
print("accuracy: ", final_r2)                                                              #85.24%


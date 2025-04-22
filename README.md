California Housing Price Prediction 🏡
This project builds a machine learning model to predict housing prices in California using various features like location, population, and income levels.
It uses data preprocessing pipelines, feature engineering, hyperparameter tuning, and a powerful model: XGBoost Regressor.

📂 Project Structure
Load Data: Read housing.csv.

Data Splitting: Split into training and test sets (stratified based on income category).

Data Visualization: Scatter plots to understand geographical and price distributions.

Correlation Analysis: Find relationships between features and the target (median_house_value).

Feature Engineering: Create new features (like rooms per household, etc.).

Data Preprocessing:

Handle missing values using SimpleImputer.

Encode categorical variables using OrdinalEncoder and OneHotEncoder.

Standardize numerical features with StandardScaler.

Build complete pipelines using Pipeline and ColumnTransformer.

Modeling:

Train a basic XGBoost Regressor model.

Hyperparameter Tuning:

Use GridSearchCV to find the best parameters for XGBoost.

Model Evaluation:

Evaluate model on both training and test sets using RMSE and R² score.

🛠️ Technologies Used
Python 3.x

Libraries:

pandas

numpy

matplotlib

scikit-learn

xgboost

📈 Final Model Performance
Training R² Score: ~0.85

Test R² Score: ~0.85

(Meaning: about 85% of variance in house prices is explained by the model!)    

📌 Key Concepts Learned
Building complete preprocessing pipelines.

Importance of stratified sampling for better generalization.

Feature engineering to boost model performance.

Using Grid Search to optimize model hyperparameters.

Applying XGBoost for efficient regression tasks.

💡 Future Improvements
Use feature selection to remove less important features.

Try Stacking, Bagging or other ensemble methods.

Deploy the model as a simple web application.

✨ Acknowledgement
Inspired by the Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow book by Aurélien Géron.

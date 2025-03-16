# Beverage-Sales

To run code:
```
streamlit run app.py
```


# Requirements:

Python==3.9.13

To install neccessary modules to run the model, use the below command
```
pip install -r requirements.txt
```

List of Modules:

- pandas==2.2.3
- numpy==1.26.4
- scikit-learn==1.2.2
- xgboost==2.0.3
- scipy==1.13.1
- streamlit


# Notebook

[Project Notebook](https://www.kaggle.com/code/abhijitrai/beverage-sales)

This notebook provides code on how model is trained and it also provides some insights on the data.

Note: Some plotly graphs are not visible in notebooks, you can rerun the notebook to display the graphs



# Model Specificaiton:

- XGBoost
- Random State = 1337
- Tree Method = "hist"
- Column sampling by Tree = 0.5
- Learning Rate = 0.1
- Max Depth of each tree = 8
- Alpha = 100
- Number of estimators(Trees) = 1000
- R square score = 92.56


# Model Compare to other type of models

DecisionTreeRegressor = 91.59
XGBoostRegressor = 91.96

Note: Currently Catboost and LGBRegressor did not support kaggle's GPU and Random Forest from scikit-learn is too slow to train in time.

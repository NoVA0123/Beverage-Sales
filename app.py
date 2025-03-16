import model_functions
import streamlit as st
from scipy import stats
import numpy as np


st.title("Beverage price predictor")
Model = model_functions.LoadTrainedModel("beverage_sales_predictor_xgb.json")
UniqueValues = model_functions.LoadUniqueValues("unique_values.json")
UniqueValues = UniqueValues["UniqueValues"]
OHEncoder = model_functions.LoadEncoder("onehot_encoder.pkl")
OrdEncoder = model_functions.LoadEncoder("ordinal_encoder.pkl")
QuantityLambda = model_functions.LoadLambda("quantity_yeo_params.json")


CustomerType = st.selectbox(
        label="Choose customer type:",
        options=UniqueValues["Customer_Type"]
        )

ProductType = st.selectbox(
        label="Choose product type:",
        options=UniqueValues["Product"]
        )

RegionType = st.selectbox(
        label="Choose Region type:",
        options=UniqueValues["Region"]
        )

CategoryType = st.selectbox(
        label="Choose category type",
        options=UniqueValues["Category"],
        index=1
        )

Quantity = st.number_input(
        label="Type value of quantity:",
        min_value=1,
        value=100,
        step=1)

QuantityValue = np.array([[Quantity]])

QuantityConvert = stats.yeojohnson(QuantityValue, lmbda=QuantityLambda)

ValuesTogetherDict = {"Customer": CustomerType,
                      "Product": ProductType,
                      "Region": RegionType,
                      "Category": CategoryType,
                      "Quantity": QuantityConvert[0][0]}

X = model_functions.Encoder(ValuesTogetherDict, OHEncoder, OrdEncoder)
YPred = Model.predict(X)

st.write("Price in Euro:", YPred[0])

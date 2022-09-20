###################
# LIBRARIES
###################

import joblib
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

###################
# FUNCTIONS
###################


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column], color="g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

house_rent_data = pd.read_csv("house_rent/House_Rent_Dataset.csv")

df = house_rent_data.copy()

# BHK: Number of Bedrooms, Hall, Kitchen.
# Rent: Rent of the Houses/Apartments/Flats.
# Size: Size of the Houses/Apartments/Flats in Square Feet.
# Floor: Houses/Apartments/Flats situated in which Floor and Total Number of Floors (Example: Ground out of 2, 3 out of 5, etc.)
# Area Type: Size of the Houses/Apartments/Flats calculated on either Super Area or Carpet Area or Build Area.
# Area Locality: Locality of the Houses/Apartments/Flats.
# City: City where the Houses/Apartments/Flats are Located.
# Furnishing Status: Furnishing Status of the Houses/Apartments/Flats, either it is Furnished or Semi-Furnished or Unfurnished.
# Tenant Preferred: Type of Tenant Preferred by the Owner or Agent.
# Bathroom: Number of Bathrooms.
# Point of Contact: Whom should you contact for more information regarding the Houses/Apartments/Flats.

check_df(df)

df.columns

# Formatting the date column to proper type
df["Posted On"] = pd.to_datetime(df["Posted On"], format="%Y/%m/%d")

df["Posted Month"] = df["Posted On"].dt.month

df.drop("Area Locality", axis=1, inplace=True)

df["Floor"] = df["Floor"].apply(lambda x: x.replace("out of", "/"))
df["Floor"] = df["Floor"].apply(lambda x: x.replace("Ground", "0"))
df["Floor"] = df["Floor"].apply(lambda x: x.replace("Lower Basement", "-1"))
df["Floor"] = df["Floor"].apply(lambda x: x.replace("Upper Basement", "-2"))
df["Floor"] = df["Floor"].apply(lambda x: x.replace(" ", ""))
df[["Current_Floor", "Max_Level"]] = df["Floor"].str.split("/", expand=True)
df.drop("Floor", axis=1, inplace=True)

# df["Size_In_Sq_Meter"] = df["Size"].apply(lambda x: x / 10.764)

df.info()

cat_cols = [col for col in df.columns if df[col].dtypes == "object"]
num_cols = [col for col in df.columns if df[col].dtypes == "int64" and str(col) != "Rent"]

for col in cat_cols:
    print("Col name: " + str(col))
    target_summary_with_cat(df, "Rent", col)


for col in num_cols:
    print("Col name: " + str(col))
    target_summary_with_num(df, col, "Rent")

for col in num_cols:
    plt.figure(figsize=(9, 6))
    g = sns.distplot(x=df[col], kde=False, color="orange", hist_kws=dict(edgecolor="black", linewidth=2))
    g.set_title("Variable: " + str(col))
    g.xaxis.set_minor_locator(AutoMinorLocator(2))
    g.yaxis.set_minor_locator(AutoMinorLocator(2))
    g.tick_params(which="both", width=2)
    g.tick_params(which="major", length=7)
    g.tick_params(which="minor", length=4)
    plt.show()

plt.figure(figsize=(9, 6))
g = sns.distplot(x=df["Rent"], kde=False, color="green", hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Rent")
g.xaxis.set_minor_locator(AutoMinorLocator(2))
g.yaxis.set_minor_locator(AutoMinorLocator(2))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show()

df.info()

df["Max_Level"] = pd.to_numeric(df["Max_Level"], errors="coerce")
df["Building Type"] = ""

df.loc[df["Max_Level"] == 1, "Building Type"] = "detached"
df.loc[(df["Max_Level"] > 1) & (df["Max_Level"] <= 5), "Building Type"] = "normal"
df.loc[(df["Max_Level"] > 5) & (df["Max_Level"] <= 11), "Building Type"] = "big"
df.loc[(df["Max_Level"] > 11), "Building Type"] = "residence"

df.groupby("Building Type").size()

df["Current_Floor"] = pd.to_numeric(df["Current_Floor"], errors="coerce")
df["Floor Level"] = ""

## Thought about taking the difference between max_level and current_floor however
## it may not be representative of how high the floor is (e.g. if I live at 25th floor and max_level is 25 also
## the diff will be 0 which may result in incorrect classification. So, I only consider the "current_floor"

df["Current_Floor"].value_counts()

df.loc[(df["Current_Floor"] > 0) & (df["Current_Floor"] < 4), "Floor Level"] = "Low"
df.loc[(df["Current_Floor"] >= 4) & (df["Current_Floor"] < 8), "Floor Level"] = "Mid"
df.loc[(df["Current_Floor"] >= 8) & (df["Current_Floor"] < 15), "Floor Level"] = "High"
df.loc[df["Current_Floor"] >= 15, "Floor Level"] = "Residence"


df.loc[df["Current_Floor"] == -1, "Floor Level"] = "Lower Basement"
df.loc[df["Current_Floor"] == -2, "Floor Level"] = "Upper Basement"
df.loc[df["Current_Floor"] == 0, "Floor Level"] = "Ground"


df["Floor Level"].value_counts().sum()


df.info()

cat_cols = [col for col in df.columns if df[col].dtypes == "object"]
num_cols = [col for col in df.columns if df[col].dtypes == "int64" and str(col) != "Rent"]

for col in cat_cols:
    print("Col name: " + str(col))
    target_summary_with_cat(df, "Rent", col)


for col in num_cols:
    print("Col name: " + str(col))
    target_summary_with_num(df, col, "Rent")

df.drop(df.loc[df["Area Type"] == "Built Area", :].index, inplace=True)
df.drop(df.loc[df["Point of Contact"] == "Contact Builder", :].index, inplace=True)
df.drop(df.loc[df["Building Type"] == "", :].index, inplace=True)
df.drop(df.loc[df["Floor Level"] == "Lower Basement", :].index, inplace=True)
df.drop(df.loc[df["Floor Level"] == "Upper Basement", :].index, inplace=True)

df.head()

plt.figure(figsize=(9, 6))
g = sns.distplot(x=np.log1p(df["Rent"]), kde=False, color="green", hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Rent")
g.xaxis.set_minor_locator(AutoMinorLocator(2))
g.yaxis.set_minor_locator(AutoMinorLocator(2))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show(block=True)

df.info()

plt.figure(figsize=(9, 6))
g = sns.distplot(x=np.log1p(df["Size"]), kde=False, color="green", hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Size")
g.xaxis.set_minor_locator(AutoMinorLocator(2))
g.yaxis.set_minor_locator(AutoMinorLocator(2))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show(block=True)


df["Rent"] = np.log1p(df["Rent"])
df["Size"] = np.log1p(df["Size"])

df["Rent"]


np.log1p(df["Size"]).describe([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T

len(df.loc[df["Size"] <= 5.5, :].index) / len(df)

df.drop(df.loc[df["Size"] <= 5.5, :].index, inplace=True)

plt.figure(figsize=(9, 6))
g = sns.distplot(x=df["Rent"], kde=False, color="green", hist_kws=dict(edgecolor="black", linewidth=2))
g.set_title("Rent")
g.xaxis.set_minor_locator(AutoMinorLocator(2))
g.yaxis.set_minor_locator(AutoMinorLocator(2))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show(block=True)

len(df.loc[df["Rent"] < 8, :].index) / len(df)
len(df.loc[df["Rent"] > 12.5, :].index) / len(df)

df.drop(df.loc[df["Rent"] < 8, :].index, inplace=True)
df.drop(df.loc[df["Rent"] > 12.5, :].index, inplace=True)

plt.figure(figsize=(9, 6))
g = sns.scatterplot(x=df["Max_Level"], y=df["Rent"], color="blue", edgecolor="black")
g.set_title("Rent vs Current_Floor")
g.xaxis.set_minor_locator(AutoMinorLocator(2))
g.yaxis.set_minor_locator(AutoMinorLocator(2))
g.tick_params(which="both", width=2)
g.tick_params(which="major", length=7)
g.tick_params(which="minor", length=4)
plt.show(block=True)

# Maybe max_level and current_floor might be interactive because of what I have stated above while handling
# the "current floor"

df["Max_Current_Level_Ratio"] = df["Max_Level"] / df["Current_Floor"]

## One Hot Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = one_hot_encoder(df, cat_cols, drop_first=True)
###################
# MODELLING
###################

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate

df.info()
X = df.drop(["Posted On", "Rent"], axis=1)
y = np.expm1(df["Rent"])

## When the model is deployed, we can predict a single value as logarithmic then apply exponential transformation

lgbm = LGBMRegressor(random_state=42)

np.mean(np.sqrt(-cross_val_score(lgbm,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))

y.describe([.01, .05, .15, .25, .50, .75, .85, .95, .99]))

###################################

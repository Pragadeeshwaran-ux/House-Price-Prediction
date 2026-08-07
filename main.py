import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("data/train.csv")

print(df.head())
df.info()
print(df.describe())

missing = df.isnull().sum()   # Create the variable

print(missing[missing > 0])   # Now it works

df = df.drop(columns=[
    "Alley",
    "PoolQC",
    "Fence",
    "MiscFeature"
])

print("Columns removed successfully!")
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"] = df["MasVnrArea"].fillna(df["MasVnrArea"].median())
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["GarageYrBlt"].median())

text_columns = [
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Electrical",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond"
]

for col in text_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

    print(df.isnull().sum())

    features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "YearRemodAdd",
    "GarageArea",
    "TotRmsAbvGrd",
    "Fireplaces"
]

X = df[features]
y = df["SalePrice"]

print(X.head())
print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

predictions = model.predict(X_test)

print("Predicted Prices:")
print(predictions[:5])

print("Actual Prices:")
print(y_test.head())

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
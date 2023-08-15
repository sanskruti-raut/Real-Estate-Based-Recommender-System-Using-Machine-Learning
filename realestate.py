import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


melb_data = pd.read_csv("melb_data.csv")
print(melb_data.columns)
y = melb_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[melbourne_features]
print(X.describe())
print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# Specify Model
melb_model = DecisionTreeRegressor(random_state=1)
# Fit Model
melb_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = melb_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
melb_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
melb_model.fit(train_X, train_y)
val_predictions = melb_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

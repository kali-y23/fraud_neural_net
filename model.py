import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("creditcard.csv")
df.shape  # (284807, 31)

# Mini-visualizations
plt.hist(df["Class"])
plt.show()

plt.hist(df["Amount"])
plt.show()

# Feature Scaling
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

scaled_features = df.copy()
col_names = ['Time', 'Amount']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

scaled_features[col_names] = features
df = scaled_features
# _________________


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
sample_size = X.shape[0]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

y_test = y_test.reshape(1, y_test.shape[0])
y_train = y_train.reshape(1, y_train.shape[0])


# GENERATING A TRAIN SET WITH A BALANCED NUMBER OF FRAUD AND NON-FRAUD TRANSACTIONS
# Counting the amount of fraud transactions
counter = 0
for entry in df["Class"]:
    if entry == 1:
        counter += 1


"""
Number of fraud transactions = counter = 492
Approx. 80% of fraudulent transactions will go to the X_balanced_train, 30% will be left for the X_balanced_test,
i.e. 392 fraud transactions in the train set and 100 in the test set
Now it is important to choose the right ratio between fraud/non-fraud in the X_balanced_train
"""

fraud_num = 392  #
# Changing this number will change the ratio between fraud/non-fraud in the X_balanced_train
ratio = 0.4  # fraud == 40% of X_balanced_train
non_fraud_num = int(fraud_num * (1 - ratio) / ratio)


# Saving all fraud/non-fraud indices into arrays
fraud_indices = []
non_fraud_indices = []

for i in range(0, 284807):
    if df["Class"][i] == 1:
        fraud_indices.append(i)
    else:
        non_fraud_indices.append(i)


# Entries with indices from these two arrays will go to the TRAIN set
random_fraud_sample = np.random.choice(fraud_indices, size=fraud_num, replace=False)
random_non_fraud_sample = np.random.choice(non_fraud_indices, size=non_fraud_num, replace=False)


# Splitting the original dataset into
X_balanced_train = np.zeros((fraud_num + non_fraud_num, 30))
X_balanced_test = np.zeros((sample_size - (fraud_num + non_fraud_num), 30))
y_balanced_train = np.zeros((1, fraud_num + non_fraud_num))
y_balanced_test = np.zeros((1, sample_size - (fraud_num + non_fraud_num)))

train_counter = 0
test_counter = 0

for i in range(0, 284807):
    if i in random_fraud_sample or i in random_non_fraud_sample:
        y_balanced_train[0][train_counter] = y[i]
        for j in range(0, 30):
            X_balanced_train[train_counter][j] = X[i][j]
        train_counter += 1
    else:
        y_balanced_test[0][test_counter] = y[i]
        for j in range(0, 30):
            X_balanced_test[test_counter][j] = X[i][j]
        test_counter += 1


#Training the net
from neural_net import DeepNeuralNet
from sklearn.metrics import confusion_matrix

nn = DeepNeuralNet((30, 1), X_train.T, y_train, print_cost=True)
costs, params = nn.train()
nn.plot_costs(costs)
pred_train = nn.predict(X_train.T, y_train, params)
pred_test = nn.predict(X_test.T, y_test, params)

confusion_matrix(np.squeeze(y_train), np.squeeze(pred_train))
confusion_matrix(np.squeeze(y_test), np.squeeze(pred_test))



nn = DeepNeuralNet((30, 1), X_balanced_train.T, y_balanced_train, print_cost=True, learning_rate=0.01, num_iterations=2500)
costs, params = nn.train()
nn.plot_costs(costs)
pred_train = nn.predict(X_balanced_train.T, y_balanced_train, params)
pred_test = nn.predict(X_balanced_test.T, y_balanced_test, params)

confusion_matrix(np.squeeze(y_balanced_train), np.squeeze(pred_train))
confusion_matrix(np.squeeze(y_balanced_test), np.squeeze(pred_test))


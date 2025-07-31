from sklearn.datasets import make_classification
import matplotlib.pylab as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

n_samples = 100

X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, random_state=250)

# Gaussian naive bayes
model = GaussianNB()

model.fit(X, y)

print(model.predict_proba(X).round(4)[:5]) # shows the probabilities that the algorithm assigns to each class after fitting

pred = model.predict(X) # based on the probabilities, predicts the binary classes for the data set

print(accuracy_score(y, pred)) # calculates the accuracy score given the predicted values'

Xc = X[y == pred] # selects the correct predictions and plots them
Xf = X[y != pred] # selects the false predictions and plots them

# Logistic regression
model = LogisticRegression(C=1, solver='lbfgs')

model.fit(X, y)

model.predict_proba(X).round(4)[:5]

pred = model.predict(X)

accuracy_score(y, pred)

Xc = X[y == pred]
Xf = X[y != pred]

plt.figure(figsize=(10,6))
plt.scatter(x=Xc[:, 0], y=Xc[:, 1], c=y[y==pred], marker='o', cmap='coolwarm')
plt.scatter(x=Xf[:, 0], y=Xf[:,1], c=y[y!= pred], marker='x', cmap='coolwarm')
plt.show()
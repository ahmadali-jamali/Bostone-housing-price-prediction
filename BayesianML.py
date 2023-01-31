# Importing modules that are required
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge

# Loading dataset
def bayes(X,y):
   

    # Splitting dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

    # Creating and training model
    model = BayesianRidge()
    model.fit(X_train, y_train)

    # Model making a prediction on test data
    prediction = model.predict(X_test)

    # Evaluation of r2 score of the model against the test set
    print(f"r2 Score Of Test Set : {r2_score(y_test, prediction)}")

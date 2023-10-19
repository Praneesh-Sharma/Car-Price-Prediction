import pickle

def predict(data):
    with open("LinearRegressionModel.pkl", 'rb') as f:
      clf = pickle.load(f)
    return clf.predict(data)

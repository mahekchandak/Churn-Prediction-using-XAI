from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

def train_models(X_train, Y_train):
    models = {}

    # Random Forest Classifier
    rfcla = RandomForestClassifier(n_estimators=100, random_state=9, n_jobs=-1)
    rfcla.fit(X_train, Y_train)
    models['Random Forest'] = rfcla

    # Naive Bayes Classifier
    nbcla = GaussianNB()
    nbcla.fit(X_train, Y_train)
    models['Naive Bayes'] = nbcla

    # Support Vector Machine
    svm_classifier = LinearSVC(random_state=42, dual=False)
    svm_classifier.fit(X_train, Y_train)
    models['SVM'] = svm_classifier

    return models
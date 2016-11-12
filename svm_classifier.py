from sklearn.multiclass import OneVsOneClassifier
def train_classifier(clf,X_train,y_train,X_test,y_test):
	clf = OneVsOneClassifier(clf)
	clf.fit(X_train, y_train)
	train_time = time() - t0
	print("train time: %0.3fs" % train_time)
	t0 = time()
	return clf
def predict_sentiment(clf,test_vector):
	pred = clf.predict(X_test)
	return pred



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, ConfusionMatrixDisplay, roc_curve, auc, r2_score
from sklearn.model_selection import train_test_split

visual_df = pd.read_csv('./src/data/finger_chords.csv')
print(visual_df)
visual_df = visual_df.dropna()

visual_df = visual_df.drop(['filename', 'time'], axis=1)
print(visual_df)

x = visual_df.drop(visual_df.columns[4], axis=1)
print(x)

y = visual_df['chord']

scale = StandardScaler()
x = scale.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

cml = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cml, display_labels=knn.classes_)
disp.plot()
plt.show()

mod_randFor = RandomForestClassifier()
mod_randFor.fit(x_train, y_train)
pred_r = mod_randFor.predict(x_test)
cmr = confusion_matrix(y_test, pred_r)
disp = ConfusionMatrixDisplay(confusion_matrix=cmr, display_labels=mod_randFor.classes_)
disp.plot()
plt.show()

mod_naive = GaussianNB()
mod_naive.fit(x_train, y_train)
pred_n = mod_naive.predict(x_test)
cmn = confusion_matrix(y_test, pred_n)
disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=mod_naive.classes_)
disp.plot()
plt.show()

mod_svm = SVC(probability = True)
mod_svm.fit(x_train,y_train)
pred_s = mod_svm.predict(x_test)
cms = confusion_matrix(y_test, pred_s)
disp = ConfusionMatrixDisplay(confusion_matrix=cms, display_labels=mod_svm.classes_)
disp.plot()
plt.show()


print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))

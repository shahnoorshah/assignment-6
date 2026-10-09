import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
iris=load_iris()
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
depths=[2,3,5,None]
for depth in depths:
    model=DecisionTreeClassifier(criterion="gini",max_depth=depth,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    print("Maximum Depth: ",depth)
    print("Testing accuracy: ",round(accuracy,4))
    print("-"*30)
    plt.figure(figsize=(10,5))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True)
    plt.title(f"Decision Tree visualization (max depth={depth})")
    plt.show()
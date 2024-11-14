import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print(mlflow.get_tracking_uri())
print(mlflow.set_tracking_uri("http://127.0.0.1:5000"))
# Load Wine data set
wine =load_wine()
x= wine.data
y= wine.target

#train test split
X_train, X_test, Y_train, Y_test= train_test_split(x,y,test_size=0.10,random_state=42)

# Define the parameter for RF Model
max_depth=15
n_estimater=10

with mlflow.start_run():
    rf= RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimater,random_state=42)
    rf.fit(X_train,Y_train)
    y_pred= rf.predict(X_test)

    accuracy= accuracy_score(Y_test,y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("max_depth", max_depth)
    mlflow.log_metric("n_estimater", n_estimater)
  
# Creating a confution matrics 

    cm= confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion matrix")

    #Save plot

    plt.savefig("Confusion_Merrix.png")

    # Log Artifact
    mlflow.log_artifact("Confusion_Merrix.png")
    mlflow.log_artifact(__file__)
    print(accuracy)



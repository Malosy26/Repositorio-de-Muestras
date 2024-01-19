#----------------------Liberias de utilidad y necesarias--------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#----------------------------------------------------------------------------------#
#-----------------------------Utiles-----------------------------------------------#
from sklearn.model_selection import train_test_split #division del train test
from sklearn.preprocessing import MinMaxScaler # escalador para normalizar os datos
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV # para realizar validacion cruzada
#----------------------------------------------------------------------------------#
#-------------------------Metricas de Clasificadores-------------------------------#
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
#----------------------------------------------------------------------------------#
#--------------------------Modelos de clasificacion--------------------------------#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#----------------------------------------------------------------------------------#
#------------------------------H2O-------------------------------------------------#
import h2o
from h2o.automl import H2OAutoML
#----------------------------------------------------------------------------------#
#-------------------------------Post-Modelo----------------------------------------#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#-----------------------------------------------------------------------------------#



def InfoImports()->str:
    importadas="""
#----------------------Liberias de utilidad y necesarias--------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#----------------------------------------------------------------------------------#
#-----------------------------Utiles-----------------------------------------------#
from sklearn.model_selection import train_test_split #division del train test
from sklearn.preprocessing import MinMaxScaler # escalador para normalizar os datos
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV # para realizar validacion cruzada
#----------------------------------------------------------------------------------#
#-------------------------Metricas de Clasificadores-------------------------------#
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
#----------------------------------------------------------------------------------#
#--------------------------Modelos de clasificacion--------------------------------#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#----------------------------------------------------------------------------------#
#------------------------------H2O-------------------------------------------------#
import h2o
from h2o.automl import H2OAutoML
#----------------------------------------------------------------------------------#
#-------------------------------Post-Modelo----------------------------------------#
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#-----------------------------------------------------------------------------------#
"""
    return importadas



def InfoClasificacion()->str:
    
    info= """
VecinosKneighbors --> KNeighborsClassifier
RegresionLogistica --> LogisticRegression
nearestCentroid --> NearestCentroid
ArbolDecision --> DesicionTreeClassifier
ArbolRandom --> RandomForestClassifier

---------- Parametros---------------
Importante: Para todas los parametros son en comun y en el mismo orden solo en ArbolDecision obtiene un ultimo parametro
adicional que es el min_samples

Todas las funciones menos la comentada anteriormente van a trabajar con sus modelos por defecto
Orden de parametros
X
y
normalizar : booleano que indica si quieres normalizar los datos
stratify : booleano que indica si quieres hacer que los datos esten stratify en y
testSize : proporcion del codigo utilizada para el test
randomState : randomState utilizado para la creacion del X_test ect

    
    """

    return info


def VecinosKneighbors(X,y,normalizar,stratify,testSize,randomState):
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state =randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state =randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
       
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
        print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        
    else:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
         model = KNeighborsClassifier(n_neighbors = 3)
         model.fit(X_train, y_train)
         yhat = model.predict(X_test)
         print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
         print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
         print("Exactitud:"    , accuracy_score(y_test, yhat))
         print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
         print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
         print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
         print("ROC AUC:", roc_auc_score(y_test, yhat))



def RegresionLogistica(X,y,normalizar,stratify,testSize,randomState):
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
        
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        print("ROC AUC:", roc_auc_score(y_test, yhat))
       
    else:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
         model = LogisticRegression()
         model.fit(X_train, y_train)
         yhat = model.predict(X_test)
         print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
         print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
         print("Exactitud:"    , accuracy_score(y_test, yhat))
         print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
         print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
         print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
         print("ROC AUC:", roc_auc_score(y_test, yhat))
        
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

    
def nearestCentroid(X,y,normalizar,stratify,testSize,randomState):
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = NearestCentroid()
        model.fit(X_train, y_train)  
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state =randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = NearestCentroid()
        model.fit(X_train, y_train)  
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
      
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = NearestCentroid()
        model.fit(X_train, y_train)  
        yhat = model.predict(X_test)
        print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
 
        print("ROC AUC:", roc_auc_score(y_test, yhat))
      
    else:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
         model = NearestCentroid()
         model.fit(X_train, y_train)  
         yhat = model.predict(X_test)
         print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
         print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
         print("Exactitud:"    , accuracy_score(y_test, yhat))
         print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
         print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
         print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
         print("ROC AUC:", roc_auc_score(y_test, yhat))
    

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
    
    
def ArbolDecision(X,y,normalizar,stratify,testSize,randomState,min_samples):
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
 
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))

    else:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
         model = DecisionTreeClassifier(min_samples_leaf=min_samples)
         model.fit(X_train, y_train) 
         yhat = model.predict(X_test)
         print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
         print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
         print("Exactitud:"    , accuracy_score(y_test, yhat))
         print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
         print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
         print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
         print("ROC AUC:", roc_auc_score(y_test, yhat))    

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

    
    
def ArbolRandom(X,y,normalizar,stratify,testSize,randomState):
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = RandomForestClassifier()
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =testSize, random_state = randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = RandomForestClassifier()
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
     
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = RandomForestClassifier()
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        print("Exactitud:"    , accuracy_score(y_test, yhat))
        print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        print("ROC AUC:", roc_auc_score(y_test, yhat))
       
    else:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
         model = RandomForestClassifier()
         model.fit(X_train, y_train) 
         yhat = model.predict(X_test)
         print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
         print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
         print("Exactitud:"    , accuracy_score(y_test, yhat))
         print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
         print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
         print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
         print("ROC AUC:", roc_auc_score(y_test, yhat))        
    
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
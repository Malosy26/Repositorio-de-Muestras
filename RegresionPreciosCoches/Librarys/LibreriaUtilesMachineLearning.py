#'--------------------------------------------------------'
#Librerias graficas y de utilidad
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
#'---------------------------------------------------------'
#Normalizacion
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
#'---------------------------------------------------------'
#Librerias para dataframe
import pandas as pd
#'---------------------------------------------------------'
#Librerias para cluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#--------------------------Modelos de clasificacion--------------------------------#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#----------------------------------------------------------------------------------#
#---------------------------Modelos de regresion-----------------------------------#


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#---------------------------------------------------------------------------------
#------------------------------Metricas regresion---------------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#--------------------------------------------------------------------------------
#'---------------------------------------------------------'
#Metricas
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

#'---------------------------------------------------------'
#Probar modelo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#'---------------------------------------------------------'



def infoImports()->str:
    
    info="""    #'--------------------------------------------------------'
                #Librerias graficas y de utilidad
                import seaborn as sns
                import matplotlib.pyplot as plt
                import numpy as np
                from collections import Counter
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import label_binarize
                from imblearn.over_sampling import SMOTE
                #'---------------------------------------------------------'
                #Normalizacion
                from sklearn.preprocessing import MinMaxScaler
                from sklearn.preprocessing import LabelEncoder
                #'---------------------------------------------------------'
                #Librerias para dataframe
                import pandas as pd
                #'---------------------------------------------------------'
                #Librerias para cluster
                from sklearn.cluster import DBSCAN
                from sklearn.cluster import KMeans

                #--------------------------Modelos de clasificacion--------------------------------#
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.neighbors import RadiusNeighborsClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.neighbors import NearestCentroid
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.ensemble import AdaBoostClassifier
                #----------------------------------------------------------------------------------#
                #'---------------------------------------------------------'
                #Metricas
                from sklearn.metrics import jaccard_score
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import precision_score
                from sklearn.metrics import recall_score
                from sklearn.metrics import f1_score
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import cross_val_score

                #'---------------------------------------------------------'
                #Probar modelo
                from sklearn.metrics import confusion_matrix
                from sklearn.metrics import classification_report
                from sklearn.metrics import roc_curve
                from sklearn.metrics import auc
                #'---------------------------------------------------------'
    """
    
    return info



def infoLibreriaUtiles()->str:
    info="""
            Cada funcion tiene individualmente un info"nombremetodo" que explica como usar la funcion esto es una
            informacion general sobre las funciones definidas
            
            -----------------------------------------EDA------------------------------------------------------------------
            correlacion()-> Funcion para imprimir la correlacion de un df
            getLabelEncoder()-> Funcion que te devuelve el objeto labelencoder incializado
            smoteDf()->funcion  que realiza el oversampling y te devuelve el df oversampleado
            
            ---------------------------------------------------------------------------------------------------------------
            
            
            ---------------------------------------Clusters----------------------------------------------------------------
            --DBSCAN--
            buscaClusterDbscan()->Funcion que retorna un df con la informacion de clusters outliers eps
            get_dbscan()-> te devuelve un string con la creacion del objeto dbscan que debes copiar y pegar en tu codigo
            giveDbscan()-> te devuelve el modelo creado 
            --DBSCAN--
            
            
            
            
            
            --KMEANS--
            inercias()-> funcion que imprime la grafica de codo por pantalla
            numOptimo()->Funcion que te devuelve aproximadamente el numero optimo de clusters
            giveKmeans()->Te devuelve el modelo KMEANS inicializado
            get_Kmeans()-> te imprime por pantalla el modelo para ponerlo en tu codigo
            
            --KMEANS--
            
            
            ---------------------------------------------------------------------------------------------------------------
            
            
            ---------------------------------------Clasificadores----------------------------------------------------------------
            
            VecinosKneighbors --> KNeighborsClassifier
            RegresionLogistica --> LogisticRegression
            nearestCentroid --> NearestCentroid
            ArbolDecision --> DesicionTreeClassifier
            ArbolRandom --> RandomForestClassifier
            Cada metodo nombrado realiza el clasificador asignado y te devuelve un diccionario con el modo
            y con las metricas para mas informacion  InfoClasificacion()
            
            df_clasificacion()-> funcion que te devuelve un DataFrame con los datos de los modelos de clasificacion probados
            comprobacion_clasificacion()->funcion para comparar las predicciones entre el conjunto de entrenamiento y el de test
            tambien te devuelve las metricas del modelo que recibe por parametro
            features()->funcion que te imprime por pantalla la grafica de la importancia de las columnas
            
            
            
            
            

            
            
            
    
    
    
    
    
    
    """
    return info

#_________________________-----------EDA--------------______________________
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def infoCorrelacion():
    info="""
            correlacion(df,X,Y) 
            El df es el dataframe que queramos ver la correlacion
            La X es para ajustar el ancho de la figura 
            La Y es para ajustar el alto de la figura
            La funcion no devuelve nada imprime por pantalla la correlacion
    
    """
    print(info)
    
def getLabelEncoder():
    label_encoder = LabelEncoder()
    return label_encoder






def correlacion(df,X,Y):
    
    corr = df._get_numeric_data().corr()  # matriz de correlación

    fig, ax = plt.subplots(figsize=(X, Y))

    mask = np.triu(np.ones_like(corr, dtype=bool))  # máscara para la matriz triangular superior

    # mapa de color coolwarm
    color_map = "coolwarm"

    # mapa de calor de correlación
    sns.heatmap(corr,                      # datos
                mask=mask,                 # máscara blanca
                cmap=color_map,            # color
                vmax=1,                    # borde vertical
                center=0,                  # centro del gráfico
                square=True,               # representación cuadrada de los datos
                linewidth=.5,              # ancho de línea
                cbar_kws={'shrink': .5},   # barra lateral de la leyenda
                annot=True,                # valor de correlación
                fmt=".2f",
                ax=ax                      # ejes para el tamaño del gráfico
               )

    plt.show()
    
def info_smoteDf()->str:
    info="""
            smoteDf(X,y,target,sampling)
            X e y son las variables
            target es el nombre de la columna a predecir
            sampling es el numero que quieres usar para oversamplear
            devuelve el df con el oversampling realizado
    
    
    """
    return info
    
def smoteDf(X,y,target,sampling):
    
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar SMOTE solo al conjunto de entrenamiento para evitar información del conjunto de prueba
    smote = SMOTE(sampling_strategy=sampling,random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Crear un nuevo DataFrame con los datos balanceados
    df_resampled = pd.DataFrame(X_train_resampled, columns=X.columns)
    df_resampled[target] = y_train_resampled
    return df_resampled
    

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------




#______________________________---------CLUSTERS---------______________________________

#---------------------------------------DBSCAN-----------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
    
def info_clusterDbscan():
    info="""
            def buscaClusterDbscan(X,inicio,fin,salto):
            La funcion recibe los siguientes parametros
            X = la variable sin normalizar
            inicio,fin,salto  son numeros float o enteros 
            
            El objetico de la funcion es obtener el eps optimo para el problema que desees resolver
            se realiza un bucle que iterara desde inicio hasta fin de salto en salto
            es decir si inicio es 0.01 fin es 2 y salto es 0.01 
            eps para 0.01
            eps para 0.02
            ......
            hasta 
            eps para 0.2
            
            La funcion devuelve un dataframe con las siguientes columnas
            eps    total_cluster   outliers  info_cluster
            
            eps es el numero de eps en esa operacion
            total_cluster es el numero de clusters
            outliers es el numero de outliers
            info_cluster es una lista de listas cada lista dentro de esta lista tiene el 2 numeros el primero es el 
            numero del cluster y el segundo la cantidad de objetos que alberga
    
    
    """
    return info
    
    
def buscaClusterDbscan(X,inicio,fin,salto):
    datos = []
    # 0.001, 2, 0.001
    for eps in np.arange(inicio, fin, salto):
    
        dbscan = DBSCAN(eps=eps, min_samples=X.shape[1]+1)
        dbscan.fit(X)

        etiquetas = dbscan.labels_

        total_clusters = len(set(etiquetas))

        # Puedes guardar el contador de etiquetas en cada iteración
        etiquetas_contador = Counter(etiquetas)

        # Puedes agregar información sobre cada cluster en el DataFrame
        info_clusters = [(cluster, count) for cluster, count in etiquetas_contador.items()]

        outlier = etiquetas_contador[-1] if -1 in etiquetas_contador else 0

        # Descomenta si deseas realizar alguna acción basada en total_clusters
        # if total_clusters > 1:
        #     ...

        datos.append([eps, total_clusters, outlier, info_clusters])

    columnas_df = ["eps", "total_clusters", "outlier", "info_clusters"]
    df_dbscan = pd.DataFrame(data=datos, columns=columnas_df)
    return df_dbscan


def get_dbscan()->str:
    r="""
    
    dbscan = DBSCAN(eps = 1.416	, min_samples = X.shape[1]+1)
    dbscan.fit(X)
    
    
    
    
    """
    return r

def info_get_dbscan()->str:
    info="""
    
        El get_dbscan() te devuelve un string para que copies la creacion del objeto dbscan en tu codigo
        esta funcion se hace por si quieres hacer pruebas o variaciones con mas parametros del dbscan
    
    
    
    
    """
    return info

def giveDbscan(X,eps):
    dbscan = DBSCAN(eps = eps	, min_samples = X.shape[1]+1)
    dbscan.fit(X)
    return dbscan

def info_giveDbscan()->str:
    info="""
            Esta funcion te devuelve el objeto dbscan realizado con min_samples X.shape[1]+1
            y los eps que recibe por parametro
    """
#---------------------------------------DBSCAN-----------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

#---------------------------------------Kmeans-----------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

def infoInercias()->str:
    info="""
            Recibe por parametro la X y calcula las inercias y te imprime por pantalla el codo para 
            que puedas comprobar visualmente el numero optimo de clusters
    
    
    """
    return info


def inercias(X):
    
    x_scaler = MinMaxScaler()
    X = x_scaler.fit_transform(X)
    inercias = list() 

    for k in range(1, 11): 
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)     
        inercias.append(kmeans.inertia_) 
        
    plt.figure(figsize = (10, 8))
    plt.plot(range(1, len(inercias) + 1), inercias, marker = "x", color = "blue")
    plt.xlabel("K's") 
    plt.ylabel("Inercia") 
    plt.show()
    
def infoNumOptimo()->str:
    info="""
            Te devuelve segun las diferenciales el calculo del numero de cluster optimo
    
    
    """
    return info
    
def numOptimo(X)->int:
    # Lista para guardar los valores de SSE
    sse = []

    # Rango de valores de K para probar
    k_values = range(1, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=123)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Calcular la tasa de cambio de SSE
    delta_sse = np.diff(sse)

    # Encontrar donde la tasa de cambio es mínima (punto de inflexión)
    optimal_k = np.argmin(delta_sse) + 2  # +2 porque los índices comienzan en 0 y hemos diferenciado el array

    return optimal_k


def info_giveKmeans():
    info="Te devuelve el objeto kmeans entrenado con X para k clusters y con random_state=123"
    return info

def giveKmeans(X,k):
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit_predict(X)
    return kmeans

def get_kmeans():
    r="""
            kmeans = KMeans(n_clusters=k, random_state=123)
            kmeans.fit_predict(X)
    """
    
    return r




#---------------------------------------Kmeans-----------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------




#---------------------------------------Clasificadores-----------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------


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
multi : es un booleano para saber si la clasificacion es multiclase
    
    """

    return info



def VecinosKneighbors(X,y,normalizar,stratify,testSize,randomState,multi):
    dic_metric={}
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state =randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        #if multi:
            #dic_metric['ROC AUC']=roc_auc_score(y_test,yhat,multiclass='ovr')
         # else:
         #    dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='Kneighbors'
        dic_metric['Tipo']='MODELO NORMALIZADO STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
        
    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state =randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='Kneighbors'
        dic_metric['Tipo']='MODELO NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
       
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        # print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        dic_metric['Nombre']='Kneighbors'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        model = KNeighborsClassifier(n_neighbors = 3)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='Kneighbors'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
            



def RegresionLogistica(X,y,normalizar,stratify,testSize,randomState,multi):
    dic_metric={}
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='LogisticRegression'
        dic_metric['Tipo']='MODELO  NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='LogisticRegression'
        dic_metric['Tipo']='MODELO  NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
        
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='LogisticRegression'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
       
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='LogisticRegression'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
        
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

    
def nearestCentroid(X,y,normalizar,stratify,testSize,randomState,multi):
    dic_metric={}
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = NearestCentroid()
        model.fit(X_train, y_train)  
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='Centroid'
        dic_metric['Tipo']='MODELO NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state =randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = NearestCentroid()
        model.fit(X_train, y_train)  
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='Centroid'
        dic_metric['Tipo']='MODELO NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
      
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = NearestCentroid()
        model.fit(X_train, y_train)  
        yhat = model.predict(X_test)
#         print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
#         print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
#         print("Exactitud:"    , accuracy_score(y_test, yhat))
#         print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
#         print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
#         print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
 
#         print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='Centroid'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
      
    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        model = NearestCentroid()
        model.fit(X_train, y_train)  
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='Centroid'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
    

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
    
    
def ArbolDecision(X,y,normalizar,stratify,testSize,randomState,min_samples,multi):
    dic_metric={}
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='DecisionTree'
        dic_metric['Tipo']='MODELO NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Min_samples']=min_samples
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='DecisionTree'
        dic_metric['Tipo']='MODELO NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Min_samples']=min_samples
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
 
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='DecisionTree'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Min_samples']=min_samples
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
      

    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        model = DecisionTreeClassifier(min_samples_leaf=min_samples)
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))    
        dic_metric['Nombre']='DecisionTree'
        dic_metric['Tipo']='MODELO  SIN NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Min_samples']=min_samples
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

    
    
def ArbolRandom(X,y,normalizar,stratify,testSize,randomState,multi):
    dic_metric={}
    if(normalizar)and(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = RandomForestClassifier()
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='RandomForest'
        dic_metric['Tipo']='MODELO NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric

    elif(normalizar)and not(stratify):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =testSize, random_state = randomState)
        scaler = MinMaxScaler()
        scaler.fit(X_train)#aplicamos el scaler a xtrain y con el scaler resultante transformamos xtrain y xtest
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.transform(X_test)
        model = RandomForestClassifier()
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='RandomForest'
        dic_metric['Tipo']='MODELO  NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
     
    elif(stratify)and not(normalizar):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState, stratify=y)
        model = RandomForestClassifier()
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO CON STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='RandomForest'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO CON STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
       
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)
        model = RandomForestClassifier()
        model.fit(X_train, y_train) 
        yhat = model.predict(X_test)
        # print(f'MODELO SIN NORMALIZADO SIN STRATIFY EN Y')
        # print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
        # print("Exactitud:"    , accuracy_score(y_test, yhat))
        # print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
        # print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
        # print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
        # print("ROC AUC:", roc_auc_score(y_test, yhat))
        dic_metric['Nombre']='RandomForest'
        dic_metric['Tipo']='MODELO SIN NORMALIZADO SIN STRATIFY EN Y'
        dic_metric['Jaccard index']=jaccard_score(y_test, yhat, average = "macro")
        dic_metric['Exactitud']=accuracy_score(y_test, yhat)
        dic_metric['Presicion']=precision_score(y_test, yhat, average = "macro")
        dic_metric['Sensibilidad']=recall_score(y_test, yhat, average = "macro")
        dic_metric['F1-score']= f1_score(y_test, yhat, average = "macro")
        if multi:
            y_test_bin = label_binarize(y_test, classes=model.classes_)
            yhat_bin = label_binarize(yhat, classes=model.classes_)
            dic_metric['ROC AUC']=roc_auc_score(y_test_bin,yhat_bin,multi_class='ovr')
        else:
            dic_metric['ROC AUC']= roc_auc_score(y_test, yhat)
        return dic_metric
        
def info_df_clasificacion()->str:
    info="""
    Funcion que recibe por parametros la X la y el ultimo parametro es si la clasifiacion es multiple o no
    multi=1 clasificacion multiple 
    multi=0 clasificacion binaria
    
    
    
    """
    return info
    
def df_clasificacion(X,y,multi):
    lista_dicc=[]
    lista_dicc.append(VecinosKneighbors(X,y,1,1,0.2,42,multi))
    lista_dicc.append(VecinosKneighbors(X,y,0,1,0.2,42,multi))
    lista_dicc.append(VecinosKneighbors(X,y,1,0,0.2,42,multi))
    lista_dicc.append(VecinosKneighbors(X,y,0,0,0.2,42,multi))

    lista_dicc.append(RegresionLogistica(X,y,1,1,0.2,42,multi))
    lista_dicc.append(RegresionLogistica(X,y,0,1,0.2,42,multi))
    lista_dicc.append(RegresionLogistica(X,y,1,0,0.2,42,multi))
    lista_dicc.append(RegresionLogistica(X,y,0,0,0.2,42,multi))

    lista_dicc.append(nearestCentroid(X,y,1,1,0.2,42,multi))
    lista_dicc.append(nearestCentroid(X,y,0,1,0.2,42,multi))
    lista_dicc.append(nearestCentroid(X,y,1,0,0.2,42,multi))
    lista_dicc.append(nearestCentroid(X,y,0,0,0.2,42,multi))

    lista_dicc.append(ArbolDecision(X,y,1,1,0.2,42,4,multi))
    lista_dicc.append(ArbolDecision(X,y,0,1,0.2,42,4,multi))
    lista_dicc.append(ArbolDecision(X,y,1,0,0.2,42,4,multi))
    lista_dicc.append(ArbolDecision(X,y,0,0,0.2,42,4,multi))

    lista_dicc.append(ArbolRandom(X,y,1,1,0.2,42,multi))
    lista_dicc.append(ArbolRandom(X,y,0,1,0.2,42,multi))
    lista_dicc.append(ArbolRandom(X,y,1,0,0.2,42,multi))
    lista_dicc.append(ArbolRandom(X,y,0,0,0.2,42,multi))
    
    columnas = ['Nombre', 'Tipo', 'Jaccard index', 'Exactitud', 'Presicion', 'Sensibilidad', 'F1-score', 'ROC AUC']
    df_clasificacion=pd.DataFrame(columns=columnas)
    df_clasificacion.head(1)
    for modelo in lista_dicc:
        df_clasificacion = pd.concat([df_clasificacion, pd.DataFrame(modelo, index=[0])], ignore_index=True)
    return df_clasificacion

def info_comprobacion_clasificacion()->str:
    info="""
    comprobacion_clasificacion(modelo,X,y,stratify)
    modelo-> es el modelo que quieres comprobar
    X e y son las variables
    stratify es por si quieres que al hacer la division en train y en test haga stratify en y
    
    
    """
    return info

def comprobacion_clasificacion(modelo,X,y,stratify):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify=stratify)
    yhat = modelo.predict(X_test)
    y_train= model_regularized.predict(X_train)
    y_test = model_regularized.predict(X_test)
    f1_train = f1_score(y_train, y_train_pred_regularized, average='macro')
    f1_test = f1_score(y_test, y_test_pred_regularized, average='macro')
    print(f'F1-score en Conjunto de Entrenamiento con max_depth = 3: {f1_train}')
    print(f'F1-score en Conjunto de Prueba con max_depth=3: {f1_test}')
    print("----------------------------------------------------------------")
    
    print("Todas las metricas")
    print("Jaccard Index:", jaccard_score(y_test, yhat, average = "macro"))
    print("Exactitud:"    , accuracy_score(y_test, yhat))
    print("Precisión:"    , precision_score(y_test, yhat, average = "macro"))
    print("Sensibilidad:" , recall_score(y_test, yhat, average = "macro"))
    print("F1-score:"     , f1_score(y_test, yhat, average = "macro"))
    print("ROC AUC:", roc_auc_score(y_test, yhat))
    
def info_features()->str:
    info="""
          features(modelo,X_train)
          Funcion que recibe el modelo y X_train y te imprime por pantalla 
          la importancia de las columnas
    """
    return info



def features(modelo,X_train):
    feature_importances = modelo.feature_importances_
    feature_names = X_train.columns  
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Visualizar la importancia de las características
    plt.figure(figsize=(12, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Características')
    plt.ylabel('Importancia')
    plt.title('Importancia de las Características en Random Forest')
    plt.xticks(rotation=45, ha='right')
    plt.show()
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------
#----------------------------   REGRESION------------------------------------------------
#-----------------------------------------------------------------------------------------------



# Función para entrenar y evaluar un modelo de regresión lineal
def train_linear_regression(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Linear Regression Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión polinomial
def train_polynomial_regression(X_train, X_test, y_train, y_test, degree=2):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Polynomial Regression (degree={degree}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión Ridge
def train_ridge_regression(X_train, X_test, y_train, y_test, alpha=1.0):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Ridge Regression (alpha={alpha}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión Lasso
def train_lasso_regression(X_train, X_test, y_train, y_test, alpha=1.0):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Lasso Regression (alpha={alpha}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión ElasticNet
def train_elasticnet_regression(X_train, X_test, y_train, y_test, alpha=1.0, l1_ratio=0.5):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"ElasticNet Regression (alpha={alpha}, l1_ratio={l1_ratio}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión KNN
def train_knn_regression(X_train, X_test, y_train, y_test, n_neighbors=5):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"KNN Regression (n_neighbors={n_neighbors}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión de Árboles de Decisión
def train_decision_tree_regression(X_train, X_test, y_train, y_test, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Decision Tree Regression (max_depth={max_depth}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión de Bosques Aleatorios
def train_random_forest_regression(X_train, X_test, y_train, y_test, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest Regression (n_estimators={n_estimators}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")

# Función para entrenar y evaluar un modelo de regresión de Máquinas de Vectores de Soporte (SVR)
def train_svr_regression(X_train, X_test, y_train, y_test, kernel='rbf', C=1.0, epsilon=0.1):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"SVR Regression (kernel={kernel}, C={C}, epsilon={epsilon}) Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    print("---------------------------------")
    
def train_polynomial_regression(X_train, X_test, y_train, y_test,degree=2):

    
    # Crear el modelo de regresión polinomial
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calcular métricas de evaluación
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Imprimir las métricas
    print("Métricas del modelo de regresión polinomial:")
    print(f"Grado del polinomio: {degree}")
    print(f"Error cuadrático medio (MSE) - Conjunto de entrenamiento: {mse_train:.2f}")
    print(f"Error cuadrático medio (MSE) - Conjunto de prueba: {mse_test:.2f}")
    print(f"Error absoluto medio (MAE) - Conjunto de entrenamiento: {mae_train:.2f}")
    print(f"Error absoluto medio (MAE) - Conjunto de prueba: {mae_test:.2f}")
    print(f"Coeficiente de determinación (R^2) - Conjunto de entrenamiento: {r2_train:.2f}")
    print(f"Coeficiente de determinación (R^2) - Conjunto de prueba: {r2_test:.2f}")
















#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix,precision_recall_fscore_support,classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

#for retrieving input and target columns,spliting to train-test and scaling the data
def preprocess_split(data):
    x = data.drop(['quality'], axis = 1)
    y = data['quality']
    y = LabelEncoder().fit_transform(y)
    #Spliting data to train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    # Feature Scaling to X_train and X_test to classify better.
    x_train = StandardScaler().fit_transform(x_train) #TO CHECK
    x_test = StandardScaler().fit_transform(x_test) #TO CHECK
    return x_train, x_test, y_train, y_test

#evaluating the SVM scores
def count_metrics(data):
    x_train, x_test, y_train, y_test=preprocess_split(data)
    ## Fitting Kernel SVM to the Training set
    #rbf works better than other classifiers
    classifier = SVC(kernel = 'rbf', decision_function_shape='ovo')
    classifier.fit(x_train, y_train)
    #Predicting the Test Set
    y_pred = classifier.predict(x_test)
    #confusion matrix of every class 
    #cm = multilabel_confusion_matrix(y_pred,y_test)
    #prin(cm)
    #micro to take into consideration all classes together
    f1 =f1_score(y_test, y_pred,average='weighted', labels=np.unique(y_pred))
    precision = precision_score(y_test, y_pred,average='weighted')
    recall = recall_score(y_test, y_pred,average='weighted')
    print("Precision Score : ",precision)
    print("Recall Score : ",recall)
    print("f1 Score : ",f1)
    print(precision_recall_fscore_support(y_test, y_pred,average='weighted'))
    return f1,precision,recall

    #linear (0.625, 0.625, 0.625)
    #rbf (0.5725, 0.5725, 0.5725)
    #rbf with scaler (0.635, 0.635, 0.635)
    
#replacing the 33% of pH column of the data set with Nan
def diminish_set(full_data):
    ph = full_data['pH'].values
    rowIndices = np.random.choice(len(full_data['pH'].values),int(len(full_data['pH'].values)/3),replace=False)
    for i in rowIndices:
        ph[i]=np.nan
    full_data.update(ph)
    return full_data

#selection of 4 different ways to handle the missing values
def missing_values_handler(method,data):    
    if method=='0':
         my_set = data
    elif method=='1':
        diminished_set = diminish_set(data)
        my_set=delete_column(diminished_set)
    elif method =='2':
        diminished_set = diminish_set(data)
        my_set=fill_with_mean(diminished_set)
    elif method =='3': 
        diminished_set = diminish_set(data)
        my_set=Logistic_regression(diminished_set)
    elif method =='4': 
        col_names=data.columns
        prev_centroids,prev_labels = k_means_to_full_set(data)
        diminished_set = diminish_set(data)
        my_set=k_means_missing(diminished_set,6, prev_centroids,prev_labels,col_names)
        
    return my_set
    
#METHODS to handle missing values
def delete_column(data):  
    #DELETE COLUMN OF pH
    data=data.drop('pH',axis=1)
    return data

def fill_with_mean(data):
    #FILL NAN WITH MEAN OF pH COLUMN
    #data.fillna(data.mean(), inplace=True)
    av = data['pH'].mean()
    data['pH'].values[np.isnan(data['pH'].values)]= av
    return data


def Logistic_regression(data):
    #FILL NAN WITH LOGISTIC REFRESSION   
    print("Not implemented")
    pass

#finding the  best number of clusters for the problem
def optimal_number_of_clusters(data):
    #Find the number of clusters
    wcss = []
    for i in range (1,7): #6 cluster
        kmeans = KMeans(n_clusters = i, init='k-means++', random_state=0) 
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)     
    
    plot.plot(range(1,7),wcss)
    plot.title('Elbow Method')
    plot.xlabel('Number of clusters')
    plot.ylabel('wcss')
    plot.show()
    best_num_K=3
    return best_num_K

#perform kmeans to find the first centroids and labels
def k_means_to_full_set(data):
    km=KMeans(n_clusters=6).fit(data)
    centroids = km.cluster_centers_
    labels = km.labels_
    return centroids,labels

#fill missing values (Nans) with Kmeans methdo
def k_means_missing(diminished_set,n_clusters, prev_centroids,prev_labels,col_names):
    #1.drop pH column
    #find clusters without pH
    #find mean of clusters in pH
    #fill values with these means
    missing = ~np.isfinite(diminished_set)
    mu = np.nanmean(diminished_set, 0, keepdims=1)
    new_data = np.where(missing, mu, diminished_set)
    max_iter=10
    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            method = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            method = KMeans(n_clusters, n_jobs=-1)

        # perform clustering on the filled-in data
        labels = method.fit_predict(new_data)
        centroids = method.cluster_centers_
        # fill in the missing values based on their cluster centroids

        new_data[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

    prev_labels = labels
    prev_centroids = method.cluster_centers_
    new_data = pd.DataFrame (new_data, columns = col_names) #make it Dataframe
    return new_data

def main():    
    data = pd.read_csv('winequality-red.csv')
    method = input('''What method you want to implement?'
                   0.Do nothing(use full pH column)
                   1.Delete column
                   2.Fill with mean
                   3.Logistic Regression
                   4.K Means
                   ''')
    #create new set depending on the method of handling the missing values
    my_set = missing_values_handler(method,data) 
    #count f1,precision,recall and print them inside count_metrics function
    f1,precision,recall=count_metrics(my_set)    

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np

from tqdm import tqdm
from utils import fetch_data


if __name__ == '__main__':
    
    labels, lesions, patients = fetch_data()
    
    lesions_agg = lesions.groupby('gpcr_id').agg({
        'voxels': np.sum,
        'max_suv_val': np.mean,
        'mean_suv_val': np.mean,
        'min_suv_val': np.mean,
        'sd_suv_val': np.mean,
        'assigned_organ': pd.Series.tolist
    }).reset_index()

    dataset = lesions_agg.merge(patients, on='gpcr_id', how='inner')
    dataset.set_index('gpcr_id', inplace=True)
    
    from utils import Preprocessor

    # Separate features by type
    numerical = list(dataset.select_dtypes(np.number).columns)
    categorical = list(dataset.select_dtypes([bool, object]).columns)
    multivalue = ['assigned_organ', 'immuno_therapy_type']

    # Remove multivalue features from categorical ones
    for feature in multivalue:
        categorical.remove(feature)
        
    features_range = list(range(len(numerical) + len(categorical) + len(multivalue)))
    bp = np.cumsum([len(numerical), len(categorical), len(multivalue)])

    # Build PipeLine of ColumnTransformers
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    ct = Pipeline([
        ('imputers', ColumnTransformer([
            ('median', SimpleImputer(strategy='median'), numerical),
            ('frequent', SimpleImputer(strategy='most_frequent'), categorical)
        ], remainder='passthrough')),
        ('preprocess', ColumnTransformer([
            ('scaler', StandardScaler(), features_range[0:bp[0]]),
            ('one-hot', OneHotEncoder(handle_unknown='ignore'), features_range[bp[0]:bp[1]]),
            ('count-vec1', CountVectorizer(analyzer=set), features_range[bp[1]:bp[2]][0]),
            ('count-vec2', CountVectorizer(analyzer=set), features_range[bp[1]:bp[2]][1])
        ], remainder='passthrough')),
    ])

    ppor = Preprocessor(
        pipe=ct,
        feats_out_fn=lambda ct: ct.named_steps['imputers'].transformers_[0][2] \
        + list(ct.named_steps['preprocess'].transformers_[1][1].get_feature_names()) \
        + ct.named_steps['preprocess'].transformers_[2][1].get_feature_names() \
        + ct.named_steps['preprocess'].transformers_[3][1].get_feature_names())

    from sklearn.model_selection import train_test_split

    I_train, I_test, y_train, y_test = \
        train_test_split(labels.index, labels, test_size=0.2, random_state=27)
        
    ppor.fit(dataset.loc[I_train])

    X_train = ppor.transform(dataset.loc[I_train])
    X_test = ppor.transform(dataset.loc[I_test])

    y_train = labels.loc[I_train]
    y_test = labels.loc[I_test]
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "Logistic Regression"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=7),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=.01, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        LogisticRegression(penalty='l2', solver='liblinear')]
    
    import mlflow
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    for name, clf in tqdm(zip(names, classifiers), total=len(names)):
        
        mlflow.start_run(run_name=name)

        mlflow.log_param('Model', name)

        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        bin_class_metrics = precision_recall_fscore_support(y_test, y_pred, average='binary')
        for value, metric in zip(list(bin_class_metrics)[:-1], ['precision', 'recall', 'fscore']):
            mlflow.log_metric(metric.capitalize() + ' - testing', value)
        
        mlflow.log_metric('Accuracy - testing', accuracy_score(y_test, y_pred))
        mlflow.log_metric('ROC AUC - testing', roc_auc_score(y_test, y_pred))
        
        mlflow.end_run()

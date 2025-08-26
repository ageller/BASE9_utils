import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import arviz as az
from diptest import diptest
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pickle


def create_features(directory, column=0, max_nfiles=np.inf, file_prefix="NGC_2682_", file_suffix="",ess_num_samples=10000):  
    '''
    function that will calculate all the features needed for the ML model 
    Note that the file names for the res files must contain the ids (and can include a prefix and suffix)

    inputs:
    - directory : (string) path to the data directory that contains the res files from BASE-9
    - column : (string) column number to use from the res file to use to calculate features
    - max_nfiles : (int) maximum number of files to use
    - file_prefix : (string) prefix in the res file names before the id 
    - file_suffix : (string) suffix in the res file names after the id
    - ess_num_samples : (int) number of samples to use in ess normal distribution

    outputs:
    - pandas DataFrame with the calculated features (see code for more details)
    '''
    
    def calculate_ess(mean_vals, std_dev_vals, num_samples=10000):
        '''
        function to calculate effective sample size for given
        inputs:
        - mean_vals : the expected mean values for the feature (array)
        - std_vals : the expected standard deviation values for the feature (array)
        - num_samples : number of random samples to use for calculation
        '''
        ess_values = []
        for m, s in zip(mean_vals, std_dev_vals):
            # Simulate MCMC samples for each value
            samples = np.random.normal(loc=m, scale=s, size=num_samples)
            samples_reshaped = samples[np.newaxis, :]  # Reshape for ArviZ
            ess = az.ess(samples_reshaped)  # Calculate ESS
            ess_values.append(ess)
        return ess_values

    # creating empty arrays for each feature
    Median = []
    Mean = []
    Percent16 = []
    Percent84 = []
    Width = []
    Source_id = []
    SnR = []
    Stdev = []
    dip_val = []
    dip_p = []
    ks_val = []
    ks_p = []
    upper = []
    lower = []
    
    file_count = 0

    for filename in os.listdir(directory):
        if file_count >= max_nfiles:
            break

        if filename.endswith(".res"):
            file_path = os.path.join(directory, filename)

            try:

                data = np.genfromtxt(file_path, skip_header=1, usecols=column)

                # calculating all the features
                feature_16th = np.percentile(data, 16)
                feature_84th = np.percentile(data, 84)
                feature_med = np.median(data)
                feature_mean = np.mean(data)
                feature_wid = feature_84th - feature_16th
                feature_std = np.std(data)
                # inputting nan wherever dividing by zero
                feature_snr = (feature_mean / feature_std if feature_std != 0 else np.nan) 
                Upper_bound = feature_84th - feature_med
                Lower_bound = feature_med - feature_16th

                # appending empty arrays with calculated values
                Percent16.append(feature_16th)
                Percent84.append(feature_84th)
                Median.append(feature_med)
                Width.append(feature_wid)
                Stdev.append(feature_std)
                Mean.append(feature_mean)
                SnR.append(feature_snr)
                upper.append(Upper_bound)
                lower.append(Lower_bound)
                

                # appending the source id array with sourceid value from file name
                sid = filename.replace(".res", "").replace(file_prefix, "").replace(file_suffix, "")
                Source_id.append(sid)

                # calculating dip value and p value
                # diptest tests for unimodality .. whether a data set has a single peak or multiple
                # p value indicates whether the data is likely from a unimodial distribution (~ >.05 likely )
                dip_value, p_value = diptest(data) 
                dip_val.append(dip_value)
                dip_p.append(p_value)
                
                # ks test is comparing our dataset to the example of what a normal distribution would look like with our dataset's mean age and stdev
                # p value indicates the probablity of observing the measured difference between two distributions
                # creating an example data set with the same mean feature and stdev as dataset
                normal_dist = np.random.normal(feature_mean, feature_std, len(data))  
                ks_statistic, ks_p_value = stats.kstest(data, normal_dist)
                ks_val.append(ks_statistic)
                ks_p.append(ks_p_value)
                
                file_count += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # adding column in for calculated ESS (the function expects arrays)
    ESS = calculate_ess(Mean, Stdev, num_samples=ess_num_samples)  

    # creating the DataFrame containing all the calculated features
    features_df = pd.DataFrame(
        {
            "source_id": Source_id,
            "Width": Width,
            "Upper_bound": Upper_bound,
            "Lower_bound": Lower_bound,
            "Stdev": Stdev,
            "SnR": SnR,
            "Dip_p": dip_p,
            "Dip_value": dip_val,
            "KS_value": ks_val,
            "KS_p": ks_p,
            "ESS": ESS,
        }
    )

    # removing rows with NaN in snr column
    features_df = features_df.dropna(subset=["SnR"])  

    # return the DataFrame to the use
    return features_df  


def create_model(
    features_df,
    label_df,
    label_column_name="Single Sampling",
    feature_columns=[
        "Width",
        "Upper_bound",
        "Lower_bound",
        "Stdev",
        "SnR",
        "Dip_p",
        "Dip_value",
        "KS_value",
        "KS_p",
        "ESS",
    ],
    random_seed=42
):

    '''
    function that will create a random forest model using scikit-learn

    inputs:
    - features_df : (pandas DataFrame) contains all the features needed for the model, including an ID column (e.g., from create_features function)
    - label_df : (pandas DataFrame) contains a label for each id in the features_df to train the model
    - label_column_name : (string) the name of the column in label_df that has the desired label for training
    - feature_columns : (list of strings) a list of column names in features_df to use for the model 
    - random_seed : (int) used for test_train_split and RandomForestClassifier

    outputs:
    - pipe: scikit-learn pipeline object containing the random forest model and standard scaler
    - X : (np array) every row is a different star, every column is a feature (same order as y)
    - y: (np array)  every row is a different star, every column is the label (same order as X)
    - X_train : (np array) portion of X that was used to train the data
    - y_train : (np array) portion of y used to train the data
    - X_test : (np array) portion of X that can be used to test the data (if user desires)
    - y_test : (np array) portion of y that can be used to test the data
    '''

    # making sure source_id in both dataframes are the same datatype
    features_df = features_df.assign(source_id=features_df["source_id"].astype("string"))
    label_df = label_df.assign(source_id=label_df["source_id"].astype("string"))

    # merging dataframes based on source ids to match ids to known sampling labels
    merged_df = features_df.merge(label_df[["source_id", label_column_name]], on="source_id", how="left")
    # dropping any rows with NaN values
    merged_df = merged_df.dropna(subset=[label_column_name])  

    # grab only the features and label that the user wants
    _, X = prepare_df_for_model(merged_df, feature_columns=feature_columns)
    # (we want the model to predict these labels)
    y = merged_df[label_column_name].to_numpy()  

    # Splitting the data into training and test sets
    # random_state - sets a seed in random number generator, ensures the splits generated are reproducible
    # test_size - proportion of dataset to include in test split ~ 30% of data is in test split
    # allows to train a model on training set and test its accuracy on testing set
    random_state = np.random.seed(random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # equalize the number of objects in each class for the training data
    # get the size of the smallest class
    u_classes = np.unique(y_train)
    c_min = len(y_train)
    for c in u_classes:
        foo = len(np.where(y_train == c)[0])
        if foo < c_min:
            c_min = foo
    # Get balanced indices
    balanced_indices = np.hstack(
        [
            np.random.choice(np.where(y_train == c)[0], c_min, replace=False)
            for c in u_classes
        ]
    )
    # Shuffle the balanced training set
    np.random.shuffle(balanced_indices)
    X_train = X_train[balanced_indices]
    y_train = y_train[balanced_indices]
    # final check
    for c in u_classes:
        print(
            f"There are {len(np.where(y_train == c)[0])} training elements with classification = {c}"
        )

    # create an sklearn pipeline to handle the scaling and classification
    # 
    # The StandardScaler will shift the mean and scale to unit variance
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=random_state))
    ])

    # fit to the training data
    pipe.fit(X_train, y_train)

    return pipe, X, y, X_train, y_train, X_test, y_test


def make_preds(
    pipe,
    X,
    y_test=None,
    feature_columns=[
        "Width",
        "Upper_bound",
        "Lower_bound",
        "Stdev",
        "SnR",
        "Dip_p",
        "Dip_value",
        "KS_value",
        "KS_p",
        "ESS",
    ],
):
    '''
    function that uses the model (from create_model) to generate labels on new data
    this function can alsob e used to test the quality of the model
    
    inputs:
    - pipe : scikit-learn pipeline object containing the random forest model and scaler objects (e.g., generated by create_model)
    - X : (np array, or pandas DataFrame) contains features to pass to the model.  every row is a different star, every column is a feature (same order as y)
      note: if the user passes a DataFrame, it will be converted to numpy array using prepare_df_for_model
    - y_test : (np array, optional) labels for X (in same order) that can be used to test the model
    - feature_columns : (list of strings) a list of column names in features_df to use for the model (must be the same as in create_model)
    
    outputs:
    - np array witht he predictions of the model (in same order as X)
    '''

    # if the user passes a dataframe, convert it to numpy first
    if isinstance(X, pd.DataFrame):
        _, X = prepare_df_for_model(X, feature_columns=feature_columns)

    # use the pipeline to scale the input, run through the classifier, and output the predicted labels
    y_pred = pipe.predict(X)  

    # Print evaluation metrics
    if y_test is not None:
        # fraction of predicted labels (y_pred) that match the corresponding labels in y_test
        print("Accuracy:", accuracy_score(y_test, y_pred))
        # text report showing main classification metrics
        print(classification_report(y_test, y_pred))

        # Feature importance: measures how much a feature contributes to a model's prediction
        feature_importances = pipe['rf'].feature_importances_
        important_features = pd.Series(feature_importances, index=feature_columns).sort_values(ascending=False)
        print("Feature Importance Ranking:")
        print(important_features)

    return y_pred


def prepare_df_for_model(
    df,
    feature_columns=[
        "Width",
        "Upper_bound",
        "Lower_bound",
        "Stdev",
        "SnR",
        "Dip_p",
        "Dip_value",
        "KS_value",
        "KS_p",
        "ESS",
    ],
):
    '''
    prepare a pandas DataFrame for input into make_preds function
    inputs:
    - df: (pandas DataFrame) contains all the features for the model, possibly in the wrong order (e.g., created by create_features)
    - feature_columns : (list of strings) a list of column names in features_df to use for the model (must be the same as in create_model)

    outputs:
    - df_clean : (pandas DataFrame) containing only the desired feature_columns (easier to look at by a human)
    - df_array : (np array) same data but in numpy array format, to be used with make_preds
    '''
    df_clean = df[feature_columns]
    df_array = df_clean.to_numpy()

    return df_clean, df_array


def save_model(pipe, filename="my_model.pkl"):
    '''
    saves the model to a python pickle file for later use
    
    inputs:
    - pipe : scikit-learn pipeline that contains the scaler and the model (e.g., from create_model)
    - filename : (string) name of the file to save 

    output:
    - the function will create a pickle file (named filename) that contains the pipeline object
    '''
    pickle.dump(pipe, open(filename, "wb"))


def load_model(filename="my_model.pkl"):
    '''
    loads the model from a python pickle file and returns to the user
    
    inputs:
    - filename : (string) name of the pickle file to load.  The contents of the file is expected
      to be a sklearn pipeline object
    output:
    - scikit-learn pipeline that contains the scaler and the model (e.g., from create_model)
    '''

    with open(filename, "rb") as f:
        pipe = pickle.load(f)

    return pipe
import os
import abc
import numpy as np
import pandas as pd
from pprint import pprint
from typing import List, Dict, AnyStr, Tuple, Set
from sklearn.model_selection import train_test_split


# set up pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Naïve Bayes Classifier




class ML_ModelInterface(metaclass=abc.ABCMeta):
    """Abstract base class for model creation"""

    @abc.abstractmethod
    def load_data_set(self, file_path: str, file_name: str, chunk_size: int, **columns) -> pd.DataFrame:
        """Load in the target data set"""
        pass

    @abc.abstractmethod
    def transformations(self, df: pd.DataFrame, **cols) -> List[Dict[str, float]]:
        """Create a transformations for the specified columns"""
        pass

    @abc.abstractmethod
    def categorical_encoders(self, df: pd.DataFrame, sample_size: float) -> None:
        """Set the Encoding type to perform on the categorical data"""
        pass

    @abc.abstractmethod
    def set_target_feature(self, feature: str) -> None:
        """Set the target(y) feature"""
        pass

    @abc.abstractmethod
    def feature_selection(self, df: pd.DataFrame, **measures) -> None:
        """Select the top n features based on a suite of accuracy measures"""
        pass

    @abc.abstractmethod
    def build_model(self) -> None:
        """Create a model """
        pass


class ML_Model(ML_ModelInterface):
    """Concrete implementation of ML_ModelInterface"""

    def __init__(self):
        self.data_set: pd.DataFrame = None      # stores the current chunk
        self.dummy_df: pd.DataFrame = None      # One hot encoded data frame
        self.frames: List[pd.DataFrame] = []    # stores the data set in chunks
        self.variable_mapping: dict = {}

        # model variables
        self.y: pd.Series = None  # the target variable
        self.X: pd.DataFrame = None  # the predictors
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data_set(self, file_path: str, file_name: str, chunk_size: int, **columns):
        """*required method"""
        try:
            if os.path.isdir(file_path):
                if os.path.isfile(os.path.join(file_path, file_name)):
                    current_chunk = pd.read_csv(
                        filepath_or_buffer=os.path.join(file_path, file_name),
                        low_memory=False, engine='c', chunksize=chunk_size,
                        usecols=[columns[c] for c in columns]
                    )
                    # grab only one chunk at a time
                    for chunk in current_chunk:
                        # perform data frame transformations
                        df = self.transformations(chunk, **columns)
                        self.frames.append(df)

                    # coerce into a single data frame
                    self.data_set = pd.concat(self.frames)

                    return self.data_set
                else:
                    raise OSError(f"OSError: File: {file_name} not found in directory: {file_path}.")
            else:
                raise OSError(f"OSError: Directory: {file_path} not found.")
        except OSError as e:
            print(e)

    def transformations(self, df: pd.DataFrame, **cols):
        """*required method"""

        for col in cols:
            if df[cols[col]].dtype == 'object':
                unique_values = set(df[cols[col]].values)
                d = {list(unique_values)[i]: i+1 for i, _ in enumerate(list(unique_values))}

                # maintain a dictionary of conversions
                self.variable_mapping.update({col: d})

                # if the df has NaN values, fill them
                if df[cols[col]].isna().sum() > 0:
                    df[cols[col]].fillna(value=-99)

                # transform the data frame column
                df[cols[col]] = [d[item] for item in df[cols[col]]]

        return df

    def categorical_encoders(self, df: pd.DataFrame, sample_size: float) -> pd.DataFrame:
        """*required method"""

        # make sure the sample_size is in the (0, 1] interval
        assert sample_size > 0.0 and sample_size <= 1.0

        # take a random sample of data from the data set to avoid memory limitations
        random_sample = df.sample(frac=sample_size, replace=False)

        # get a list of prefixes
        dummy_prefix = [p for p in list(random_sample.columns)]

        # One hot encode the data to build binary classifiers
        self.dummy_df = pd.concat(
            [pd.get_dummies(random_sample[col], prefix=p)
                for col in random_sample
                for p in dummy_prefix
             ],
            axis=1,
            keys=random_sample.columns
        )
        # return the dummy data frame
        return self.dummy_df

    def set_target_feature(self, feature: str) -> None:
        """*required method"""
        self.y = self.data_set[feature]
        self.X = self.data_set.drop(columns=feature)

    def feature_selection(self, df: pd.DataFrame, **measures):
        """*required method"""
        pass

    def build_model(self):
        """*required method"""
        pass




def main():

    file_path = "/Users/williamrobertmurphy/Downloads"
    file_name = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv"
    ml_model = ML_Model()
    ml_model.load_data_set(
        file_path=file_path, file_name=file_name, chunk_size=200000,
        gender='Gender', race='Race', ethnicity='Ethnicity', age_group='Age Group',
        type_of_admission='Type of Admission',
        severity_of_illness='APR Severity of Illness Description',
        risk_of_mortality='APR Risk of Mortality',
        css_procedure_description='CCS Diagnosis Description',
        css_procedure_code='CCS Procedure Code'
    )
    ml_model.set_target_feature(feature='CCS Diagnosis Description')

    encoded_df = ml_model.categorical_encoders(df=ml_model.X, sample_size=0.1)
    pprint(ml_model.y)

if __name__ == '__main__':
    main()

    
    
    import os
import abc
import numpy as np
import pandas as pd
from pprint import pprint
from typing import List, Dict, AnyStr, Tuple, Set
from sklearn.model_selection import train_test_split
from  sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier

# set up pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Use Random Forests: Better than Decision Tree(they over fit)




class ML_ModelInterface(metaclass=abc.ABCMeta):
    """Abstract base class for model creation"""

    @abc.abstractmethod
    def load_data_set(self, file_path: str, file_name: str, chunk_size: int, **columns) -> pd.DataFrame:
        """Load in the target data set"""
        pass

    @abc.abstractmethod
    def transformations(self, df: pd.DataFrame, **cols) -> List[Dict[str, float]]:
        """Create a transformations for the specified columns"""
        pass

    @abc.abstractmethod
    def categorical_encoders(self, df: pd.DataFrame, sample_size: float, feature: str) -> None:
        """Set the Encoding type to perform on the categorical data"""
        pass

    @abc.abstractmethod
    def set_target_feature(self, df: pd.DataFrame, feature: str) -> None:
        """Set the target(y) feature"""
        pass

    @abc.abstractmethod
    def feature_selection(self, df: pd.DataFrame, k: int) -> None:
        """Select the top n features based on a suite of accuracy measures"""
        pass

    @abc.abstractmethod
    def build_model(self) -> None:
        """Create a model """
        pass


class ML_Model(ML_ModelInterface):
    """Concrete implementation of ML_ModelInterface"""

    def __init__(self):
        self.data_set: pd.DataFrame = None      # stores the current chunk
        self.dummy_df: pd.DataFrame = None      # One hot encoded data frame
        self.frames: List[pd.DataFrame] = []    # stores the data set in chunks
        self.variable_mapping: dict = {}

        # model variables
        self.y: pd.Series = None  # the target variable
        self.X: pd.DataFrame = None  # the predictors
        self.selected_features: pd.DataFrame = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data_set(self, file_path: str, file_name: str, chunk_size: int, **columns):
        """*required method"""
        try:
            if os.path.isdir(file_path):
                if os.path.isfile(os.path.join(file_path, file_name)):
                    current_chunk = pd.read_csv(
                        filepath_or_buffer=os.path.join(file_path, file_name),
                        low_memory=False, engine='c', chunksize=chunk_size,
                        usecols=[columns[c] for c in columns]
                    )
                    # grab only one chunk at a time
                    for chunk in current_chunk:
                        # perform data frame transformations
                        df = self.transformations(chunk, **columns)
                        self.frames.append(df)

                    # coerce into a single data frame
                    self.data_set = pd.concat(self.frames)

                    return self.data_set
                else:
                    raise OSError(f"OSError: File: {file_name} not found in directory: {file_path}.")
            else:
                raise OSError(f"OSError: Directory: {file_path} not found.")
        except OSError as e:
            print(e)

    def transformations(self, df: pd.DataFrame, **cols):
        """*required method"""

        for col in cols:
            if df[cols[col]].dtype == 'object':
                unique_values = set(df[cols[col]].values)
                d = {list(unique_values)[i]: i+1 for i, _ in enumerate(list(unique_values))}

                # maintain a dictionary of conversions
                self.variable_mapping.update({col: d})

                # if the df has NaN values, fill them
                if df[cols[col]].isna().sum() > 0:
                    df[cols[col]].fillna(value=-99)

                # transform the data frame column
                df[cols[col]] = [d[item] for item in df[cols[col]]]

        return df

    def categorical_encoders(self, df: pd.DataFrame, sample_size: float, feature: str) -> pd.DataFrame:
        """*required method"""

        # make sure the sample_size is in the (0, 1] interval
        assert sample_size > 0.0 and sample_size <= 1.0

        # take a random sample of data from the data set to avoid memory limitations
        random_sample = df.sample(frac=sample_size, replace=False)

        # set the target and predictors
        self.set_target_feature(df=random_sample, feature=feature)

        # get a list of prefixes
        dummy_prefix = [p for p in list(random_sample.columns)]

        # One hot encode the data to build binary classifiers
        self.dummy_df = pd.concat(
            [pd.get_dummies(random_sample[col], prefix=p)
                for col in random_sample
                for p in dummy_prefix
             ],
            axis=1,
            keys=random_sample.columns
        )

        # set the predictors
        self.X = self.dummy_df.drop(columns=feature)

        # return the dummy data frame
        return self.dummy_df

    def set_target_feature(self, df: pd.DataFrame, feature: str) -> None:
        """*required method"""
        self.y = df[feature]

    def feature_selection(self, df: pd.DataFrame, k: int):
        """*required method"""

        feature = SelectKBest(f_classif, k=k)
        X_new = feature.fit(df, self.y)

        scores = pd.DataFrame()
        scores['F score'] = feature.scores_
        scores['P value'] = feature.pvalues_
        scores['Support'] = feature.get_support()
        scores['Attribute'] = df.columns

        selected_features = scores[(scores.Support == True)]
        attributes = [attr.rstrip('\n') for _, attr in selected_features.Attribute]

        # build a new data frame from the selected multi-index features
        features = []
        for meta in df.columns.levels[0]:
            print(meta)
            #df.loc[:, ('Age Group', 'Age Group_1')]
            pprint(df.loc[meta].head())


        # set the selected features
        self.selected_features = pd.concat(features)
        pprint(self.selected_features.head(10))
        return  scores


    def build_model(self):
        """*required method"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(

        )




def main():

    file_path = "/Users/williamrobertmurphy/Downloads"
    file_name = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv"
    ml_model = ML_Model()
    ml_model.load_data_set(
        file_path=file_path, file_name=file_name, chunk_size=200000,
        gender='Gender', race='Race', ethnicity='Ethnicity', age_group='Age Group',
        type_of_admission='Type of Admission',
        severity_of_illness='APR Severity of Illness Description',
        risk_of_mortality='APR Risk of Mortality',
        css_procedure_description='CCS Diagnosis Description'
    )

    # select the target feature(y)
    #ml_model.set_target_feature(feature='CCS Diagnosis Description')

    # one hot encode the predictors(X)
    ml_model.categorical_encoders(df=ml_model.data_set, sample_size=0.15,
                                  feature='CCS Diagnosis Description')

    # perform feature selection
    #pprint(ml_model.X.head())
    #pprint(type(ml_model.X.loc[:, ('Age Group','Age Group_1')]))
    #output = ml_model.feature_selection(df=ml_model.X, k=4)
    #pprint(ml_model.selected_features.columns)
    pprint(list(ml_model.X.columns.levels[0]))
    pprint(list(ml_model.X.columns.levels[1]))

    #pprint(ml_model.y)
    #pprint(ml_model.dummy_df.head())
    #pprint(ml_model.dummy_df.columns)
    #pprint(ml_model.y.shape)
    #pprint(output)
if __name__ == '__main__':
    main()

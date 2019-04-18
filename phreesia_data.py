"""
model
~~~~~
Create a model for the gov.ny.health dataset.
goals:
1.) Predict a patient’s procedure given the information available
from their hospital admission (primarily patient’s demographics and chief complaint).
2.) Predict the length of stay of a patient upon admission.

# Parameter Tuining
############################################################################
# model parameters                       # model hyper-parameters
- learned or estimated from the data        - set before training occurs
- result of fitting a model                 - specify how training is supposed to happen
- used in future predictions
- not manually set

# Dimensionality Reduction
############################################################################
# feature selection                                 # feature extraction
- remove features from the data set                 - extract new features from original ones
- completely remove irrelevant features              - little irrelevant information within them


# Data Standardization
############################################################################

# Imbalanced Classes
############################################################################
# Stratified Sampling


# Feature Engineering
############################################################################
# Categorical Variables
    - One Hot Encoding: Code to 1's and 0's when there are more tan 2 variables to encode
    example:

"""
import os
import abc
import numpy as np
import pandas as pd
import pandas.errors as pd_err
import sklearn.linear_model
from typing import List, Dict, AnyStr, Tuple, Set
from pprint import pprint
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score,
                             mean_absolute_error, log_loss, classification_report)
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  Lasso, SGDClassifier, Ridge)
from sklearn.feature_selection import (VarianceThreshold, chi2, SelectKBest, RFE,
                                       SelectFromModel, f_regression, SelectPercentile,
                                       f_classif, RFECV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVR


# set up pandas display options
pd.get_option("display.max_rows", 1000)


class ModelInterface(metaclass=abc.ABCMeta):
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
    def build_model(self, ml_algo, target_feature: str, **algo_params) -> sklearn.linear_model:
        """Build a model to test"""
        pass

    @abc.abstractmethod
    def feature_selection(self, *models, **kwargs) -> None:
        """Choose the best model for the task at hand."""
        pass

    @abc.abstractmethod
    def model_selection(self, *models, **kwargs) -> None:
        """Choose the most appropriate model"""
        pass


class ConcreteModel(ModelInterface):
    """A concrete Model class built from the interface"""

    def __init__(self):
        self.model = None
        self.model_scores: dict = {}            # {model_name: score}
        self.data_set: pd.DataFrame = None      # stores the current chunk
        self.frames: List[pd.DataFrame] = []    # list of chunked data frame
        self.variable_mapping: dict = {}        # stores the transformations of variables
        self.transformation_list: list = []

        # model variables
        self.y: pd.Series = None                # the target variable
        self.X: pd.DataFrame = None             # the predictors
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test =  None

    def load_data_set(self, file_path: str, file_name: str, chunk_size: int, **columns) -> pd.DataFrame:
        """*required method
        Load in the target data set
        """
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

    def create_train_tests_sets(self, test_size_: float, shuffle_=True, stratify_=None):
        """*required
        Split the data into training and testing sets
        """

        assert test_size_ < 1.0 # test_set must be less than 1.0
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                        self.X, self.y, test_size=test_size_, shuffle=shuffle_,
                        stratify=stratify_, random_state=np.random.randint(0, 1000)
        )
        print(f"X_train.shape: {self.X_train.shape}")
        print(f"X_test.shape: {self.X_test.shape}")

    def cross_validation(self, n:int) -> None:
        """Perform cross validation"""
        kf = KFold(n_splits=n, shuffle=True)
        for train, test in kf.split(self.data_set.values):
            pass


    def transformations(self, df, **cols) -> pd.DataFrame:
        """*required
        Create a transformations for the specified columns
        """
        total_old_mem: float = 0.0
        total_new_mem: float = 0.0

        for col in cols:
            if df[cols[col]].dtype == 'object':
                unique_values = set(df[cols[col]].values)
                d = {list(unique_values)[i]: i+1 for i, _ in enumerate(list(unique_values))}

                # maintain a dictionary of conversions
                self.variable_mapping.update({col: d})

                # current size of the column in MB
                current_size = df[cols[col]].memory_usage(deep=True)/1048576
                total_old_mem += current_size
                # if the df has NaN values, fill them
                if df[cols[col]].isna().sum() > 0:
                    df[cols[col]].fillna(value=-99)

                # transform the data frame column
                df[cols[col]] = [d[item] for item in df[cols[col]]]
                new_size = df[cols[col]].memory_usage(deep=True)/1048576
                total_new_mem += new_size

                # calculate the total difference in size
                size_diff = current_size - new_size
                self.transformation_list.append({col: {'before conversion mem': current_size,
                                                  'after conversion mem': new_size,
                                                  'difference': size_diff}})
        return df

    def select_predictor_variable(self, target_feature) -> None:
        """Select the target feature, i.e. self.y"""
        self.y = self.data_set[target_feature]
        self.X = self.data_set.drop(columns=target_feature)
        #print("In select_predictor_variable")
        #print(self.y.head(10))
        #print(self.X.head(20))
        #print(self.X.columns)


    def build_model(self, ml_algo, target_feature: str, **algo_params) -> sklearn.linear_model:
        """*required
        Build a model to test
        """

        if ml_algo in list(globals().keys()):
            algo = globals().get(ml_algo)()
            #pprint(type(algo))
            y = self.data_set[target_feature]
            X = self.data_set.drop(columns=target_feature)
            #print(f"y(target feature): {y.head()}")
            #print(f"features: {X.columns}")

        # check feature independence
        features = list(X.columns)
        for feature in features:
            print(f"Spearmanr Rank\nfeatures: {target_feature} | {feature}")
            spear, p_value = spearmanr(y, X[feature])
            print(f"Spearmanr: {spear}")
            print("============\n\n")


    def feature_selection(self, *models, **kwargs):
        """*required method
        Choose the best model for the task at hand.
        χ²-test: Nonparametric test, used for categorical features

        low variance with target: we can drop them.
        forward selection: iteratively add features
        backward selection: start with all and iteratively remove
        LASSO: performs variable selection + regularization
        Trees:


        """
        chi_square_results   = []  # Chi^2 test
        rfe_results          = []  # Recursive Feature Elimination
        pca_results          = []  # Principle Component Analysis
        F_test_results       = []  # ANOVA F-value

        variance_thresh = VarianceThreshold(threshold=.8*(1-.8))
        var_fit = variance_thresh.fit(self.X_train)
        var_df = self.X_train[self.X_train.columns[variance_thresh.get_support(indices=True)]]
        pprint(f"Variance Threshold: {var_df}")
        pprint(f"Variance Threshold Type: {type(var_df)}")
        #print(f"Variance Threshold: {var_fit.transform(self.X)}")




        # target and predictors
        # self.y = self.data_set[target_feature]
        # pprint(f"type of y: {type(self.y)}")
        # self.X = self.data_set.drop(columns=target_feature)
        # pprint(f"type of X: {type(self.X)}")
        #
        # pprint(f"Y.shape: {self.y.shape}")
        # pprint(f"X.shape: {self.X.shape}")
        # pprint(f"Y.nan values: {self.y.isna().sum()}")
        # pprint(f"X.nan values: {self.X.isna().sum()}")

        # χ²-test
        # chi_squared_test = SelectKBest(score_func=chi2, k=4)
        # chi_squared_fit = chi_squared_test.fit(self.X_train.values, self.y_train.values)
        # print("Chi Squared Test")
        # print(chi_squared_fit.scores_)
        #
        # # # F-test
        # f_test = SelectKBest(f_classif, k=3)
        # f_test_fit = f_test.fit(self.X_train.values, self.y_train.values)
        # f_test_scores = f_test_fit.scores_
        # f_test_pvalues = f_test_fit.pvalues_
        # pprint(f"F-test Scores: {f_test_scores}")
        # pprint(f"F-test p-values: {f_test_pvalues}")

        # # Model Based Feature Selection
        # mbfs = SelectFromModel(
        #         RandomForestClassifier(n_estimators=3, random_state=np.random.randint(0, 1000)),
        #     threshold='median'
        # )
        # pprint(f"Model Based Feature Selection: {type(mbfs)}")


    def model_selection(self, *models, **kwargs):
        """*required
        Choose the most appropriate model
        """
        # TODO: Build this
        pass





        # RFE test
        # model = LogisticRegression()
        # rfe   = RFE(model, 3)
        # rfe_fit = rfe.fit(X, y)
        # print(f"Number of Features: {rfe_fit.n_features_}\n"
        #       f"Selected Features: {rfe_fit.support_}\n"
        #       f"Featrue Ranking: {rfe_fit.ranking_}")

        # PCA test
        # pca = PCA(n_components=3)
        # pca_fit = pca.fit(X)
        # print(f"Variance: {pca_fit.explained_variance_ratio_}")
        # print(f"Components: {pca_fit.components_}")





def main():
    """Run the application"""

    #file_path = "/Users/williamrobertmurphy/Downloads"
    file_path = 'N:\\USD\\Business Data and Analytics\\Will dev folder'
    file_name = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv"

    # create an instance of the ConcreteModel class
    ml_model = ConcreteModel()
    # load the data and select the features you want to use
    ml_model.load_data_set(file_path=file_path, file_name=file_name, chunk_size=100000,
                           gender='Gender', race='Race', ethnicity='Ethnicity', age_group='Age Group',
                           type_of_admission='Type of Admission',
                           severity_of_illness='APR Severity of Illness Description',
                           risk_of_mortality='APR Risk of Mortality',
                           css_procedure_description='CCS Diagnosis Description',
                           css_procedure_code='CCS Procedure Code')

    # set the target feature and predictor array
    ml_model.select_predictor_variable(target_feature='CCS Diagnosis Description')

    # build training and testing sets
    ml_model.create_train_tests_sets(test_size_=0.6)

    # perform feature selection
    ml_model.feature_selection()




    #ml_model.build_model(ml_algo='LogisticRegression', target_feature='CCS Diagnosis Description')

    # ml_model.transformations(ml_model.data_set, gender='Gender', race='Race', ethnicity='Ethnicity', age_group='Age Group',
    #                        type_of_admission='Type of Admission',
    #                        severity_of_illness='APR Severity of Illness Description',
    #                        risk_of_mortality='APR Risk of Mortality')

if __name__ == '__main__':
    main()


"""
model_build
"""
import os
import abc
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.preprocessing import OneHotEncoder


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class ML_ModelInterface(metaclass=abc.ABCMeta):
    """Abstract base class for model creation"""

    @abc.abstractmethod
    def load_dataset(self, file_path: str, file_name: str, chunk_size: int, **columns):
        """Load in the target data set"""
        pass

    @abc.abstractmethod
    def data_standardization(self, df, **cols):
        """Standardize the data for better algorithm performance"""
        pass


class ML_Model(ML_ModelInterface):
    """A concrete Model class built from the interface"""

    def __init__(self):
        self.data_set = None
        self.frames = []
        self.variable_mapping: dict = {}

    def load_dataset(self, file_path: str, file_name: str, chunk_size: int, **columns):
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
                        self.data_set = chunk
                        # perform data frame transformations
                        #self.data_set = self.data_standardization(chunk, **columns)
                        break
                        #self.frames.append(df)
                    # coerce into a single data frame
                    #self.data_set = pd.concat(self.frames)

                    return self.data_set
                else:
                    raise OSError(f"OSError: File: {file_name} not found in directory: {file_path}.")
            else:
                raise OSError(f"OSError: Directory: {file_path} not found.")
        except OSError as e:
            print(e)


    def data_standardization(self, df, **cols) -> list:

        extracted_frames = []

        for col in cols:
            if df[cols[col]].dtype == 'object':
                # if the df has NaN values, fill them
                if df[cols[col]].isna().sum() > 0:
                    df[cols[col]].fillna(value=-99)
                # perform one hot encoding for nominal features
                feature = pd.get_dummies(df[cols[col]], prefix=col)
                extracted_frames.append(feature)

        # return a new data frame from the extracted features
        return extracted_frames

    def data_transformation(self, df, **cols) -> pd.DataFrame:
        """transform the data"""
        for col in cols:
            if df[cols[col]].dtype == 'object':
                # if the df has NaN values, fill them
                if df[cols[col]].isna().sum() > 0:
                    df[cols[col]].fillna(value=-99)
                # convert to numeric value
                df[cols[col]] = df[cols[col]].map({i:k for k, i in enumerate(list(set(df[cols[col]].values)))})
        return df




def main():
    file_path = 'N:\\USD\\Business Data and Analytics\\Will dev folder'
    file_name = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv"

    ml_model = ML_Model()
    df = ml_model.load_dataset(file_path=file_path, file_name=file_name,chunk_size=100000,
                          gender='Gender', race='Race', ethnicity='Ethnicity', age_group='Age Group',
                          type_of_admission='Type of Admission',
                          risk_of_mortality='APR Risk of Mortality',
                          css_procedure_description='CCS Diagnosis Description',
                          #css_procedure_code='CCS Procedure Code'
                          )
    #pprint(df.head(10))

    #df['Gender'] = df.Gender.map({'F': 0, 'M': 1, 'U': 2})
    #df['Race'] = df.Race.map({i:k for k, i in enumerate(list(set(df.Race.values)))})
    #race = pd.get_dummies(df.Race, prefix='Race')
    # pprint(df.Race.value_counts())
    # pprint(df['Age Group'].value_counts())
    # pprint(df.Ethnicity.value_counts())
    # pprint(df['Type of Admission'].value_counts())
    # pprint(df.Gender.value_counts())
    # pprint(df['CCS Diagnosis Description'].value_counts())
    # pprint(df['APR Risk of Mortality'].value_counts())


    # onehot = OneHotEncoder(dtype=np.int64, sparse=True)
    # nominals = pd.DataFrame(
    #     onehot.fit_transform(df[[c for c in list(df.columns)]]).toarray(),
    #     columns=[[c for c in list(df.columns)]]
    # )
    #pprint(df.head())
    #pprint({i:k for k, i in enumerate(list(set(df.Race.values)))})
    #pprint(race)

    new_df = ml_model.data_transformation(
        df=df, gender='Gender', race='Race', ethnicity='Ethnicity',
        age_group='Age Group', css_procedure_description='CCS Diagnosis Description'
    )

    pprint(new_df.head())

if __name__ == '__main__':
    main()


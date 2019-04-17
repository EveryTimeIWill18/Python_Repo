"""
model
~~~~~
Create a model for the gov.ny.health dataset.

goals:
1.) Predict a patient’s procedure given the information available
from their hospital admission (primarily patient’s demographics and chief complaint).

2.) Predict the length of stay of a patient upon admission.

"""
import os
import abc
import numpy as np
import pandas as pd
import pandas.errors as pd_err
from typing import List, Dict, AnyStr, Tuple, Set
from pprint import pprint
from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score,
                             mean_absolute_error, log_loss)
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  Lasso, SGDClassifier, Ridge)
from sklearn.feature_selection import (VarianceThreshold, chi2, SelectKBest,
                                       SelectFromModel, f_regression, SelectPercentile)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    def model_selection(self, *models, **kwargs) -> None:
        """Choose the best model for the task at hand."""
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


    def model_selection(self, *models, **kwargs):
        """*required method
        Choose the best model for the task at hand.
        """
        pass


def main():
    """Run the application"""
    file_path = "N:\\USD\\Business Data and Analytics\\Will dev folder"
    file_name = "Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.csv"
    ml_model = ConcreteModel()
    ml_model.load_data_set(file_path=file_path, file_name=file_name, chunk_size=100000,
                           gender='Gender', race='Race', ethnicity='Ethnicity', age_group='Age Group',
                           type_of_admission='Type of Admission',
                           severity_of_illness='APR Severity of Illness Description',
                           risk_of_mortality='APR Risk of Mortality',
                           css_procedure_description='CCS Diagnosis Description',
                           css_procedure_code='CCS Procedure Code')

    # ml_model.transformations(ml_model.data_set, gender='Gender', race='Race', ethnicity='Ethnicity', age_group='Age Group',
    #                        type_of_admission='Type of Admission',
    #                        severity_of_illness='APR Severity of Illness Description',
    #                        risk_of_mortality='APR Risk of Mortality')

if __name__ == '__main__':
    main()




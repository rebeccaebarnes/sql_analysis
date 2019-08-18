from time import time
import pandas as pd
import numpy as np
from sql_test import sql_query

class MetricCalc():
    """docstring"""

    def __init__(self, name, source_df, profile_df):
        self.name = name
        self.source_df = source_df
        self.profile_df = profile_df
        self._start_string = "Commencing {metric} calculation..."\
            .format(metric=name)
        self._end_string = "{metric} assessment complete in \
            {duration} seconds."

    def print_duration(self, start_time):
        print(self._end_string.format(metric=self.name.title(),
                                      duration=round(time() - start_time, 2)))

    def calculation(self):
        start_time = time()
        self.profile_df[self.name] = np.nan
        self.print_duration(start_time)

"""class DataProfiler():
    '''docstring for .'''

    def __init__(self, source_df, server_type=None, query_str=None,
                 save_location=None, metrics_dict=None):
        self.source_df = source_df
        self.server_type = server_type
        self.query_str = query_str
        self.save_location = save_location
        self.metrics_dict = metrics_dict
        self._profile_df = self._create_profile_df()

    class _profile_ordinal(MetricCalc):
        def calculation(self):
            start_time = time()
            self.profile_df[self.name] = np.arange(1,
                                                   self.source_df.shape[1] + 1)
            self.print_duration(start_time)

    class _profile_dtypes(self):
        '''
        Assign the Python dtype of each column in the source DataFrame to
        the profile DataFrame.
        '''
        metric = 'data types'
        start_time = time()
        print(self._start_string.format(metric=metric))
        self._profile_df['Python Data Type'] = self.source_df.dtypes
        self._print_duration(metric, start_time)

    def _create_profile_df(self):
        '''
        Create empty DataFrame as base for data profiling of source DataFrame.
        The profile contains fields for Ordinal Position, Python Data Type,
        Count, Null Count, Percent Null, Blank Count, Minimum Value, Maximum Value,
        Mean Value, Standard Deviation, Median Value, Mode, Cardinality,
        Duplicates Count, Max String Length for the columns found in the source
        DataFrame.

        Params:
            source_df: (Pandas DataFrame) Typically read from SQL table.

        Returns:
            Empty Pandas DataFrame with row count equal to the number of columns
            in the source DataFrame.
        '''
        profile_cols = ['Ordinal Position', 'Python Data Type', 'Count',
                        'Null Count', 'Percent Null', 'Blank Count',
                        'Minimum Value', 'Maximum Value', 'Mean Value',
                        'Standard Deviation', 'Median Value', 'Mode',
                        'Cardinality', 'Duplicates Count', 'Max String Length']
        profile_index = self.source_df.columns

        return pd.DataFrame([], columns=profile_cols, index=profile_index)

    def _profile_dtypes(self):
        '''
        Assign the Python dtype of each column in the source DataFrame to
        the profile DataFrame.
        '''
        metric = 'data types'
        start_time = time()
        print(self._start_string.format(metric=metric))
        self._profile_df['Python Data Type'] = self.source_df.dtypes
        self._print_duration(metric, start_time)

    def _profile_count(self):
        '''
        Assign the row count in the source DataFrame to the profile DataFrame.
        '''
        metric='row count'
        start_time = time()
        print(self._start_string.format(metric=metric))
        self._profile_df['Count'] = self.source_df.shape[0]
        self._print_duration(metric, start_time)

    def _profile_null_count(self):
        '''
        Assign the null count of each column in the source DataFrame to
        the profile DataFrame.
        '''
        metric='null count'
        start_time = time()
        print(self._start_string.format(metric=metric))
        self._profile_df['Null Count'] = pd.isnull(self.source_df).sum()
        self._print_duration(metric, start_time)

    def _profile_null_perc(self):
        '''
        Calculate the null percentage of each column in the source DataFrame and
        assign to the profile DataFrame.
        '''
        metric='null count'
        start_time = time()
        print(self._start_string.format(metric=metric))
        self._profile_df['Percent Null'] = \
        round(self._profile_df['Null Count']/self._profile_df['Count'] * 100,
              2)
        self._print_duration(metric, start_time)

    def _profile_blank(self):
        '''
        Calculate the number of empty or whitespace-only strings in each column
        in the source DataFrame and assign to the profile DataFrame.
        '''
        start_time = time()
        print("Commencing blank count calculation...")
        for row in self._profile_df.index:
            self._profile_df.loc[row, 'Blank Count'] = \
            (self.source_df[row].astype(str).str.strip() == '').sum()
        print("Blank count calculation complete in {} seconds."\
        .format(round(time() - start_time, 2)))

    def _profile_min(self):
        '''
        Assign the minimum of each column in the source DataFrame to
        the profile DataFrame.
        '''
        start_time = time()
        print("Commencing field minimum calculation...")
        self._profile_df['Minimum Value'] = self.source_df.min()
        print("Field minimum calculation complete in {} seconds."\
        .format(round(time() - start_time, 2)))

    def _profile_max(self):
        '''
        Assign the maximum of each column in the source DataFrame to
        the profile DataFrame.
        '''
        start_time = time()
        print("Commencing field maximum calculation...")
        self._profile_df['Maximum Value'] = self.source_df.max()
        print("Field maximum calculation complete in {} seconds."\
        .format(round(time() - start_time, 2)))

    def _profile_mean(self):
        '''
        Assign the mean (to five decimal places) of each numeric column in the
        source DataFrame to the profile DataFrame.
        '''
        start_time = time()
        print("Commencing field mean calculation...")
        for row in self._profile_df.index:
            if pd.api.types.is_numeric_dtype(self.source_df[row]):
                self._profile_df.loc[row, 'Mean Value'] = \
                round(self.source_df[row].mean(), 5)
        print("Field mean calculation complete in {} seconds."\
        .format(round(time() - start_time, 2)))

    def _profile_std(self):
        '''
        Assign the standard deviation (to five decimal places) of each numeric
        column in the source DataFrame to the profile DataFrame.
        '''
        start_time = time()
        print("Commencing field standard deviation calculation...")
        for row in self._profile_df.index:
            if pd.api.types.is_numeric_dtype(self.source_df[row]):
                self._profile_df.loc[row, 'Standard Deviation'] = \
                round(self.source_df[row].std(), 5)
        print("Field mean calculation complete in {} seconds."\
        .format(round(time() - start_time, 2)))

metrics_dict = dict(ordinal_position=True,
                    python_data_type=True, count_rows=True,
                    null_count=True, percent_null=True,
                    blank_count=True, minimum_value=True,
                    maximum_value=True, mean_value=True,
                    standard_deviation=True, median_value=True,
                    mode_value=True, cardinality=True,
                    duplicates_count=True, max_string_length=True)
"""
def create_profile_df(source_df):
    '''
    Create empty DataFrame as base for data profiling of source DataFrame.
    The profile contains fields for Ordinal Position, Python Data Type, Count,
    Null Count, Percent Null, Blank Count, Minimum Value, Maximum Value,
    Mean Value, Standard Deviation, Median Value, Mode, Cardinality,
    Duplicates Count, Max String Length for the columns found in the source
    DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.

    Returns:
        Empty Pandas DataFrame with row count equal to the number of columns
        in the source DataFrame.
    '''
    profile_cols = ['Ordinal Position', 'Python Data Type', 'Count',
                    'Null Count', 'Percent Null', 'Blank Count',
                    'Minimum Value', 'Maximum Value', 'Mean Value',
                    'Standard Deviation', 'Median Value', 'Mode',
                    'Cardinality', 'Duplicates Count', 'Max String Length']
    profile_index = source_df.columns

    return pd.DataFrame([], columns=profile_cols, index=profile_index)

def profile_ordinal(source_df, profile_df):
    '''
    Assign the ordinal position of each column in the source DataFrame to
    the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Ordinal Position'] = np.arange(1, source_df.shape[1] + 1)

def profile_dtypes(source_df, profile_df):
    '''
    Assign the Python dtype of each column in the source DataFrame to
    the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Python Data Type'] = source_df.dtypes

def profile_count(source_df, profile_df):
    '''
    Assign the row count in the source DataFrame to the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Count'] = source_df.shape[0]

def profile_null_count(source_df, profile_df):
    '''
    Assign the null count of each column in the source DataFrame to
    the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Null Count'] = pd.isnull(source_df).sum()

def profile_null_perc(profile_df):
    '''
    Calculate the null percentage of each column in the source DataFrame and
    assign to the profile DataFrame.

    Params:
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Percent Null'] = round(profile_df['Null Count'] \
                                       /profile_df['Count'] * 100, 2)

def profile_blank(source_df, profile_df):
    '''
    Calculate the number of empty or whitespace-only strings in each column
    in the source DataFrame and assign to the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    for row in profile_df.index:
        profile_df.loc[row, 'Blank Count'] = \
        (source_df[row].astype(str).str.strip() == '').sum()

def profile_min(source_df, profile_df):
    '''
    Assign the minimum of each column in the source DataFrame to
    the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Minimum Value'] = source_df.min()

def profile_max(source_df, profile_df):
    '''
    Assign the maximum of each column in the source DataFrame to
    the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Maximum Value'] = source_df.max()

def profile_mean(source_df, profile_df):
    '''
    Assign the mean (to five decimal places) of each numeric column in the
    source DataFrame to the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    for row in profile_df.index:
        if pd.api.types.is_numeric_dtype(source_df[row]):
            profile_df.loc[row, 'Mean Value'] = round(source_df[row].mean(), 5)

def profile_std(source_df, profile_df):
    '''
    Assign the standard deviation (to five decimal places) of each numeric
    column in the source DataFrame to the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    for row in profile_df.index:
        if pd.api.types.is_numeric_dtype(source_df[row]):
            profile_df.loc[row, 'Standard Deviation'] = \
            round(source_df[row].std(), 5)

def profile_median(source_df, profile_df):
    '''
    Assign the median of each numeric column in the
    source DataFrame to the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    for row in profile_df.index:
        if pd.api.types.is_numeric_dtype(source_df[row]):
            profile_df.loc[row, 'Median Value'] = source_df[row].median()

def profile_mode(source_df, profile_df):
    '''
    Assign the mode of each column in the source DataFrame to
    the profile DataFrame if the column is not all NaN.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    for row in profile_df.index:
        if source_df[row].mode().shape[0] != 0:
            profile_df.loc[row, 'Mode'] = source_df[row].mode().iloc[0]

def profile_cardinality(source_df, profile_df):
    '''
    Assign the number of unique values of each column in the source DataFrame
    to the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    profile_df['Cardinality'] = source_df.nunique()

def profile_duplicates(source_df, profile_df):
    '''
    Assign the duplicate count of each column in the source DataFrame
    to the profile DataFrame.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    for row in profile_df.index:
        profile_df.loc[row, 'Duplicates Count'] = \
        source_df[row].duplicated().sum()

def profile_max_str(source_df, profile_df):
    '''
    Calculate the maximum string length in each strin column
    in the source DataFrame and assign to the profile DataFrame. NaN is
    counted as 0.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        profile_df: (Pandas DataFrame) DataFrame created by create_profile_df.
    Returns:
        None.
    '''
    for row in profile_df.index:
        if pd.api.types.is_string_dtype(source_df[row]):
            profile_df.loc[row, 'Max String Length'] = \
            source_df[row].fillna('').apply(len).max()

def create_data_profile(source_df):
    '''
    Complete a data profile of a source DataFrame, providing details on the
    following values: Ordinal Position, Python Data Type, Count,
    Null Count, Percent Null, Blank Count, Minimum Value, Maximum Value,
    Mean Value, Standard Deviation, Median Value, Mode, Cardinality,
    Duplicates Count, Max String Length.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.

    Returns:
        profile_df: (Pandas DataFrame) Pandas DataFrame with row count equal
        to the number of columns in the source DataFrame with profile details.
    '''
    print('Creating data profile...', '\n')
    profile_df = create_profile_df(source_df)

    profile_ordinal(source_df, profile_df)
    print('Ordinality assessment complete.')
    profile_dtypes(source_df, profile_df)
    print('Data type assessment complete.')
    profile_count(source_df, profile_df)
    print('Count assessment complete.')
    profile_null_count(source_df, profile_df)
    print('Null count assessment complete.')
    profile_null_perc(profile_df)
    print('Null percentage assessment complete.')
    profile_blank(source_df, profile_df)
    print('Blank count assessment complete.')
    profile_min(source_df, profile_df)
    print('Minimum value assessment complete.')
    profile_max(source_df, profile_df)
    print('Maximum value assessment complete.')
    profile_mean(source_df, profile_df)
    print('Mean value calculation complete.')
    profile_std(source_df, profile_df)
    print('Standard deviation value calculation complete.')
    profile_median(source_df, profile_df)
    print('Median value calculation complete.')
    profile_mode(source_df, profile_df)
    print('Mode value calculation complete.')
    profile_cardinality(source_df, profile_df)
    print('Cardinality assessment complete.')
    profile_duplicates(source_df, profile_df)
    print('Duplicate assessment complete.')
    profile_max_str(source_df, profile_df)
    print('Maximum string length calculation complete.', '\n')
    print('Full data profile complete.')

    return profile_df.sort_index()

def data_profile(source_df, save_location=None):
    '''
    Create and save a data profile of a source DataFrame, providing details
    on the following values: Ordinal Position, Python Data Type, Count,
    Null Count, Percent Null, Blank Count, Minimum Value, Maximum Value,
    Mean Value, Standard Deviation, Median Value, Mode, Cardinality,
    Duplicates Count, Max String Length.

    Params:
        source_df: (Pandas DataFrame) Typically read from SQL table.
        save_location: (str, optional, default=None) File directory and name
                       to save.

    Returns:
        profile_df: (Pandas DataFrame) Pandas DataFrame with row count equal
        to the number of columns in the source DataFrame with profile details.
    '''
    profile_df = create_data_profile(source_df)
    if save_location:
        profile_df.to_csv(save_location)

    return profile_df

def assess_table(db_eng_type, query_str, save_location=None):
    '''
    Complete an assessment of an Oracle database table, with server defined
    by DB_ENG, and saves the assesssment to a defined location if specified.

    Params:
        db_eng_type: (str) Key of server type from DB_ENG.
        query_str: (str) String for Oracle database query.
        save_location: (str, optional, default=None) File directory and name
                       to save.

    Returns:
        profile_df: (Pandas DataFrame) Pandas DataFrame with row count equal
        to the number of columns in the source DataFrame with profile details.
    '''
    print('Commencing query...')
    source_df = sql_query(query_str, db_eng_type)
    print('Query complete.', '\n')
    profile_df = data_profile(source_df, save_location)

    return profile_df

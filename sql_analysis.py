from datetime import datetime
import os
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sql_config_example as sqlc


DB_ENG = {'pp': sqlc.ENGINE_PP,
          'dvd': sqlc.ENGINE_DVD}

def sql_query(query_str, engine):
    '''
    Complete database query using pandas.read_sql.

    Params:
        query_str: (str) SQL query string.
        engine: (str) Server type to use from options of DB_ENG.keys().

    Returns:
        Pandas DataFrame.
    '''
    return pd.read_sql(query_str, DB_ENG[engine])

def test_input_SQLUnitTest(data, comparison_names, test_field, save_location):
    """TO DO: Add docstring"""
    # data tests
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                "The tuple contains {} values. Only 2 are accepted."\
                .format(len(data))
                )
        if not isinstance(data[0], str) or not isinstance(data[1], str):
            raise TypeError(
                "The values of the tuple must both be strings."
                )
        if data[0].split()[0].upper() != 'compare':
            raise ValueError(
                "The first value in the tuple must be a SQL compare statement."
                )
        if data[1] not in ('dev', 'prod', 'ss'):
            raise ValueError(
                "The second value in the tuple must be one of {}."\
                             .format(list(DB_ENG.keys()))
                )
    elif not isinstance(data, pd.core.frame.DataFrame):
        raise TypeError(
            "The data type for 'data' must either be a tuple"
            " or a Pandas DataFrame. The current type of 'data' is {}."\
            .format(type(data))
            )

    # comparison_names tests
    if len(comparison_names) < 2:
        raise ValueError(
            "At least two columns must be specified for comparison."
            )
    if not isinstance(comparison_names, (tuple, list)):
        raise TypeError(
            "A list-like collection must be used for 'comparison_names'."
            " A {} object was entered.".format(type(comparison_names))
            )
    col_split = data.columns.str.split('_')
    col_headers = [col[0] for col in col_split]
    col_suffixes = [col[1] for col in col_split]
    #for col in col_headers:
    #    if col not in comparison_names:
    #        raise ValueError("The prefix of each column name in 'data' must be found in 'comparison_names', followed by '_'.")
    suffix_count = 0
    for col in col_suffixes:
        if col == 'count':
            suffix_count += 1
    if suffix_count != len(comparison_names):
        raise ValueError(
            "Columns containing values counts must have the suffix '_count'."
            "There should be one '_count' column for each value in 'comparison_names'."
            )

    # test_field tests
    if test_field:
        if not isinstance(test_field, str):
            raise TypeError(
                "'test_field' must be of type str."
                )

    # save_location tests
    if save_location:
        split_test = save_location.split('\\')
        if len(split_test) > 1:
            raise ValueError(
                "Save location must use / instead of "
                "\\ to indicate sub-directories."
                )

def test_input_SQLGatherData(comparison_fields, groupby_fields,
                             table_names, table_alias, test_type):
    """TO DO: Add docstring"""
    # confirm minimum two fields
    if len(comparison_fields) < 2:
        raise ValueError(
            "Minimum length for 'comparison_fields' is 2:"
            " one field for each of at least two tables."
            )

    # confirm equal field length
    for field in (groupby_fields, table_names, table_alias):
        if len(comparison_fields) != len(field):
            raise ValueError(
                "All field lists must have the same length."
                "The length of the 'comparison_fields'"
                "does not match the length of {}.".format(field)
                )

    # confirm test_type
    test_types = ('count', 'low_distinct', 'high_distinct', 'numeric', 'id_check')
    if test_type not in test_types:
        raise ValueError(
            "The value for 'test_type' is not a valid test type."
            "Use a value from {}.".format(test_types)
            )

    # only two fields for id_check
    if test_type == 'id_check':
        if len(comparison_fields) != 2:
            raise ValueError(
                "Only two fields can be compared at a time for test type 'id_check'."
                )

class SQLUnitTest:
    """
    Complete SQL queries and equality comparisons between database tables.

    inputs:
        comparison_fields: (list-like) Name of field to be queried for each table.
                          The target table should be listed first.
        groupby_fields: (list-like) Name of field used to group for each table.
                        Field order should be consistent with that of 'comparison_fields'.
        table_names: (list-like) Name of each database table to be queried.
                     Name order should be consistent with that of 'comparison_fields'.
        table_alias: (list-like) Alias string for each database table to be queried.
                     Each alias must be a single word containing no '_'.
                     Alias order should be consistent with that of 'comparison_fields'.
        db: (str) Server alias, as specified by DB_ENG.
        test_type: {'count', 'low_distinct', 'high_distinct', 'numeric', 'id_check'}.
        save_location: (optional, str) Folder directory for saving.

    methods:
        create_test_string: Create SQL query string from specified inputs.
        gather_data: Complete database query based on specified SQL string.
        run_test: Complete the equality comparison between the '_count' columns.
        save_results: Create folder directory as needed and save results to this location.
        compare_ids: Complete a comparison of counts and id fields.
    """
    def __init__(self, comparison_fields, groupby_fields, table_names,
                 table_alias, db_server, test_type, save_location=None):
        # Test input variables
        # TODO: Manage test_input_SQLUnitTest(save_location) & SQLGatherData test

        # Convert SQL to DataFrame as needed

        self.comparison_fields = comparison_fields
        self.groupby_fields = groupby_fields
        self.table_names = table_names
        self.table_alias = table_alias
        self.db_server = db_server
        self.test_type = test_type
        self.save_location = save_location
        self._test_str = None
        self._results = None
        self._summary = pd.DataFrame([])
        self._exceptions = {}
        self._priority_review = {}
        self._today_date = datetime.today().strftime('%y%m%d')
        self._alt_date = datetime.today().strftime('%d-%b-%y')

    def _create_count_string(self):
        """TO DO: Add docstring"""
        test_str = ""

        # Create target CTE
        cte_statement = "WITH {target_alias} AS (SELECT {target_groupby}, COUNT(*) AS row_count"\
                        + " FROM {target_table} GROUP BY {target_groupby})"
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = ", {alias} AS (SELECT {groupby}, COUNT(*) AS row_count"\
                    + " FROM {table} GROUP BY {groupby})"
        for alias, groupby_field, table in zip(self.table_alias[1:],
                                               self.groupby_fields[1:],
                                               self.table_names[1:]):
            test_str += table_cte.format(alias=alias, groupby=groupby_field, table=table)

        return test_str

    def _create_low_distinct_string(self):
        """TO DO: Add docstring"""
        test_str = ""

        # Create target CTE
        cte_statement = "WITH {target_alias} AS (SELECT {target_groupby},"\
                        + " COALESCE({target_compare}, 'Unknown')"\
                        + " AS {target_compare}, COUNT(*) AS row_count"\
                        + " FROM {target_table} GROUP BY {target_groupby}, {target_compare})"
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_compare=self.comparison_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = ", {alias} AS (SELECT {groupby}, COALESCE({compare}, 'Unknown')"\
                    + " AS {compare}, "\
                    + "COUNT(*) AS row_count FROM {table} GROUP BY {groupby}, {compare})"
        for alias, groupby_field, compare_field, table in zip(self.table_alias[1:],
                                                              self.groupby_fields[1:],
                                                              self.comparison_fields[1:],
                                                              self.table_names[1:]):
            test_str += table_cte.format(alias=alias,
                                         groupby=groupby_field,
                                         compare=compare_field,
                                         table=table)

        return test_str

    def _create_high_distinct_string(self):
        """TO DO: Add docstring"""
        test_str = ""

        # Create target CTE
        cte_statement = "WITH {target_alias} AS (SELECT {target_groupby}, "\
                        + "COUNT(DISTINCT {target_compare}) AS row_count"\
                        + " FROM {target_table} GROUP BY {target_groupby})"
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_compare=self.comparison_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = ", {alias} AS (SELECT {groupby}, COUNT(DISTINCT {compare})"\
                    + " AS row_count FROM {table} GROUP BY {groupby})"
        for alias, groupby_field, compare, table in zip(self.table_alias[1:],
                                                        self.groupby_fields[1:],
                                                        self.comparison_fields[1:],
                                                        self.table_names[1:]):
            test_str += table_cte.format(alias=alias,
                                         groupby=groupby_field,
                                         compare=compare,
                                         table=table)

        return test_str

    def _create_numeric_string(self):
        """TO DO: Add docstring"""
        test_str = ""

        # Create target CTE
        cte_statement = "WITH {target_alias} AS (SELECT {target_groupby}, "\
                        + "SUM({target_compare}) AS row_count"\
                        + " FROM {target_table} GROUP BY {target_groupby})"
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_compare=self.comparison_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = ", {alias} AS (SELECT {groupby}, SUM({compare}) AS row_count"\
                    + " FROM {table} GROUP BY {groupby})"
        for alias, groupby_field, compare, table in zip(self.table_alias[1:],
                                                        self.groupby_fields[1:],
                                                        self.comparison_fields[1:],
                                                        self.table_names[1:]):
            test_str += table_cte.format(alias=alias,
                                         groupby=groupby_field,
                                         compare=compare,
                                         table=table)

        return test_str

    def _create_id_check_string(self):
        """TO DO: Create docstring"""
        target_str = "SELECT {target_groupby}, {target_compare} FROM {target_table}"\
                     .format(target_groupby=self.groupby_fields[0],
                             target_compare=self.comparison_fields[0],
                             target_table=self.table_names[0])

        source_str = ("SELECT {source_groupby}, {source_compare} AS"
                      "{target_compare} FROM {source_table}")\
                     .format(source_groupby=self.groupby_fields[1],
                             source_compare=self.comparison_fields[1],
                             target_compare=self.comparison_fields[0],
                             source_table=self.table_names[1])

        self._test_str = (target_str, source_str)

    def create_test_string(self, test_string=None):
        """
        Create SQL query string. If a test_string is provided, stores this string.
        If not, creates and stores SQL string based on values stored on instantiation.

        inputs:
            test_string: (optional, str) Custom SQL query string.
        """
        # Assign str to self._test_str
        if test_string:
            self._test_str = test_string
        elif self.test_type == 'id_check':
            self._create_id_check_string()

        # Or create initial test string
        elif self.test_type == 'count':
            test_str = self._create_count_string()
        elif self.test_type == 'low_distinct':
            test_str = self._create_low_distinct_string()
        elif self.test_type == 'high_distinct':
            test_str = self._create_high_distinct_string()
        elif self.test_type == 'numeric':
            test_str = self._create_numeric_string()

        # Create SELECT statement
        if self.test_type in ('count', 'high_distinct', 'numeric'):
            initial_select_state = (" SELECT {target_alias}.{target_groupby}, "
                                    "{target_alias}.row_count AS {target_alias}_count")
            test_str += initial_select_state.format(target_alias=self.table_alias[0],
                                                    target_groupby=self.groupby_fields[0])

        else:
            initial_select_state = (" SELECT {target_alias}.{target_groupby}, "
                                    "{target_alias}.{target_compare} AS"
                                    " {target_alias}_{target_compare}, "
                                    "{target_alias}.row_count AS {target_alias}_count")
            test_str += initial_select_state.format(target_alias=self.table_alias[0],
                                                    target_compare=self.comparison_fields[0],
                                                    target_groupby=self.groupby_fields[0])

        table_select = ", {alias}.row_count AS {alias}_count"
        for alias in self.table_alias[1:]:
            test_str += table_select.format(alias=alias)

        # Create FROM statement
        initial_from_state = " FROM {target_alias}"
        test_str += initial_from_state.format(target_alias=self.table_alias[0])

        # Create JOIN statement
        if self.test_type in ('count', 'high_distinct', 'numeric'):
            join_state = (" JOIN {alias} ON {alias}.{groupby_field} "
                          "= {target_alias}.{target_groupby}")
            for alias, groupby_field in zip(self.table_alias[1:], self.groupby_fields[1:]):
                test_str += join_state.format(alias=alias,
                                              groupby_field=groupby_field,
                                              target_alias=self.table_alias[0],
                                              target_groupby=self.groupby_fields[0])
        else:
            join_state = (" LEFT JOIN {alias} ON {alias}.{groupby_field} = "
                          "{target_alias}.{target_groupby}"
                          " AND {alias}.{compare} = {target_alias}.{target_compare}")
            for alias, groupby_field, compare_field in zip(self.table_alias[1:],
                                                           self.groupby_fields[1:],
                                                           self.comparison_fields[1:]):
                test_str += join_state.format(alias=alias,
                                              groupby_field=groupby_field,
                                              compare=compare_field,
                                              target_alias=self.table_alias[0],
                                              target_groupby=self.groupby_fields[0],
                                              target_compare=self.comparison_fields[0])

        # Add ordering
        test_str += " ORDER BY {target_alias}.{target_groupby}"\
                    .format(target_alias=self.table_alias[0],
                            target_groupby=self.groupby_fields[0])

        self._test_str = test_str

    def gather_data(self, test_string=None):
        """
        Complete query of database. If a test_string is provided, utilizes this string.
        If not,constructs and utilizes the string based on values stored on instantiation.

        inputs:
            test_string: (optional, str) Custom SQL query string.

        returns:
            result: If 'test_type' is 'id_check', a tuple of Pandas DataFrames.
                    If not, a Pandas DataFrame.
        """
        self.create_test_string(test_string)

        print('  Commencing {} query...'.format(self.comparison_fields[0]))
        if self.test_type == 'id_check':
            target_df = sql_query(self._test_str[0], self.db_server)
            source_df = sql_query(self._test_str[1], self.db_server)
            result = target_df, source_df

        else:
            result = sql_query(self._test_str, self.db_server)

        print('  Query for {} complete.'.format(self.comparison_fields[0]))
        self._results = result
        return result

    def _assess_priority_review(self, comparison_col, assess_col, review_threshold):
        """TO DO: Add docstring"""
        assessment = None
        if self._results[assess_col].mean() == 100:
            assessment = 'MISSING VALUE for ' + comparison_col + '_' + self.comparison_fields[0]
            print(assessment)

        is_not_missing = self._results[assess_col] != 100
        not_missing_median = self._results.loc[is_not_missing, assess_col].abs().median()
        if not_missing_median > review_threshold:
            assessment = 'PRIORITY REVIEW on ' + comparison_col + '_' \
                         + self.comparison_fields[0] + ': ' + str(not_missing_median)
            print(assessment)
        return assessment

    def save_results(self, index=False):
        """
        Create folder directory based on stored save_location, current date, and test_field.

        Inputs:
            index: (optional, boolean, default=False) Indicates whether the index
                   should be saved or not.
        """
        folder_name = self.save_location.strip('/') + '/' + self._today_date
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_name = '/' + self.comparison_fields[0] + '.csv'
        self._results.to_csv(folder_name + file_name, index=index)

    def run_test(self, test_string=None, review_threshold=2):
        """
        Run equality comparisons between the comparison_fields.
        If row values are not equal, the absolute difference and percentage
        difference is calculated.
        Prints and stores exceptions as they are encountered.
        Prints and stores fields flagged for priority assessment.
        Stores results in '_results'.
        Saves results if save_location is stored.

        Inputs:
            test_string: (str, optional) Custom SQL query string.
            review_threshold: (numeric, optional, default=2)
                              Percentage difference between comparison fields that
                              flags the field for priority assessment.
        """
        print('Commencing {} test for {}...'.format(self.test_type, self.comparison_fields[0]))
        self._results = self.gather_data(test_string=test_string)
        target_col = self.table_alias[0] + '_count'
        test_field = self.comparison_fields[0]
        for col in self.table_alias[1:]:
            try:
                col_name = col + '_count'
                # Get difference in counts
                compare_col = self.table_alias[0] + '_minus_' + col
                self._results[compare_col] = self._results[target_col] - self._results[col_name]

                # Get perc diff
                perc_col = 'perc_diff_' + col
                self._results[perc_col] = np.nan
                count_not_equal = self._results[compare_col] != 0
                self._results.loc[count_not_equal, perc_col] = \
                round((self._results[target_col] - self._results[col_name])\
                /self._results[target_col] * 100, 2)
                # Assess perc diff
                assessment = self._assess_priority_review(col, perc_col,
                                                          review_threshold=review_threshold)
                if assessment:
                    self._priority_review[test_field + '_' + col] = assessment
                # Assign to summary
                summary_field = self.groupby_fields[0]
                if self._summary.empty:
                    self._summary = self._results[[summary_field, perc_col]].copy()
                    self._summary.rename(columns={perc_col: test_field + '_' + col},
                                         inplace=True)
                else:
                    summary_col = self._results[[summary_field, perc_col]].copy()
                    summary_col.rename(columns={perc_col: test_field + '_' + col},
                                       inplace=True)
                    self._summary = self._summary.merge(summary_col,
                                                        how='outer',
                                                        on=summary_field)
            except Exception as e:
                print('EXCEPTION:', e)
                self._exceptions[test_field + '_' + col] = e

        # Add date
        if 'date' not in self._results.columns:
            self._results.insert(loc=0, column='date', value=self._alt_date)

        # Check save results
        if self.save_location:
            self.save_results()
        print('Test for {} complete.\n'.format(test_field))

    def compare_ids(self, table_alias, id_fields):
        """
        Complete a count comparison based on stored data and combine with a comparison of IDs.
        Stores results in '_results'.
        Saves results if save_location is stored.

        inputs:
            table_alias: (list-like) Two table alias to compare.
                         "Target" table must be listed first.
            id_fields: (list-like) Names of ID fields to compare.
                       "Target" table field must be listed first.
        """
        # Test comparison names
        if len(table_alias) != 2:
            raise ValueError(
                "For ID comparison, only two values in 'comparison_names' are accepted."
                )
        # Update for test type
        self.test_type = 'id_check'
        source_position = self.table_alias.index(table_alias[1])
        self.comparison_fields = id_fields
        self.groupby_fields = (self.groupby_fields[0], self.groupby_fields[source_position])
        self.table_names = (self.table_names[0], self.table_names[source_position])
        self.table_alias = table_alias

        target_df, source_df = self.gather_data()
        target_col = count_col = table_alias[0] + '_' + self.comparison_fields[0]
        source_col = table_alias[1] + '_' + self.comparison_fields[0]

        for col, df in zip((target_col, source_col, count_col),
                           (target_df, source_df, self._results)):
            if col not in df.columns:
                raise ValueError(
                    "Each column name in 'table_alias' must be in the respective DataFrame."
                    "{} was not found.".format(col)
                    )

        target_id = self.groupby_fields[0]
        source_id = self.groupby_fields[1]

        self._results.index = self._results[count_col]
        self._results.drop(count_col, axis=1, inplace=True)

        # Compare ids
        target_in_source_name = table_alias[0] + '_missing_in_' + table_alias[1]
        source_in_target_name = table_alias[1] + '_missing_in_' + table_alias[0]
        self._results[target_in_source_name] = np.nan
        self._results[source_in_target_name] = np.nan
        for ind in self._results.index:
            print("Commencing ID comparison for", ind, "...")
            try:
                is_ind_target = target_df[target_col] == ind
                is_ind_source = source_df[source_col] == ind
                target_in_source = target_df.loc[is_ind_target, target_id]\
                                   .isin(source_df.loc[is_ind_source, source_id])
                source_in_target = source_df.loc[is_ind_source, source_id]\
                                   .isin(target_df.loc[is_ind_target, target_id])
                self._results.loc[ind, target_in_source_name] = \
                str(target_df.loc[~target_in_source & is_ind_target, target_id].values)\
                .replace(' ', ', ')
                self._results.loc[ind, source_in_target_name] = \
                str(source_df.loc[~source_in_target & is_ind_source, source_id].values)\
                .replace(' ', ', ')
            except Exception as e:
                self._exceptions['missing_id_' + str(ind)] = e
                print('EXCEPTION missing_id', ind, ":", e)
            print('ID comparison for', ind, 'complete.')

        # Check save results
        if self.save_location:
            self.save_results(index=True)

    def summarize_results(self, summary_type='both', save='both'):
        """TODO: Add docstring"""
        # TODO: create tests for input and checking summary_field
        # TODO: manage low_distinct tests
        # TODO: manage unshared groupby fields
        summary_field = self.groupby_fields[0]
        self._summary.index = self._summary[summary_field]
        self._summary.drop(summary_field, axis=1, inplace=True)
        self._summary = self._summary.transpose()

        if save in ('data', 'both'):
            self._summary.to_csv(self.save_location + '/summary_' + self._today_date + '.csv')

        if summary_type in ('image', 'both'):
            palette = sns.diverging_palette(30, 230, s=99, l=60, n=15)
            plt.figure(figsize=(self._summary.shape[1] * 1.25, self._summary.shape[0]))
            ax = sns.heatmap(self._summary,
                             vmin=-30,
                             vmax=30,
                             cmap=palette,
                             center=0,
                             annot=True,
                             annot_kws={'fontsize': 12},
                             linewidths=1)
            ax.xaxis.tick_top()
            ax.tick_params(axis='both', which='both', length=0, rotation=0, labelsize=12)
            plt.xlabel('')
            plt.show()

            if save in ('image', 'both'):
                plt.savefig(self.save_location + '/summary_img_' + self._today_date + '.png')

        return self._summary

class MetricCalc():
    '''docstring'''

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

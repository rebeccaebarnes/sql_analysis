from datetime import datetime
import os
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sql_config as sqlc


DB_ENG = sqlc.DB_ENG

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

def test_input_string(string_var):
    # TODO: Add docstring
    """docstring"""
    if not isinstance(string_var, str):
        raise TypeError(
            "'test_string' must be of type <str>. "
            "Current type is {}".format(type(string_var))
        )

def test_input_ut_init(comparison_fields, groupby_fields,
                       table_names, table_alias, test_type,
                       db_server, save_location):
    """Test inputs for SQLUnitTest"""
    # Confirm minimum two fields
    if len(comparison_fields) < 2:
        raise ValueError(
            "Minimum length for 'comparison_fields' is 2:"
            " one field for each of at least two tables."
            )

    # Confirm equal field length
    for field, name in zip((groupby_fields, table_names, table_alias),
                           ('groupby_fields', 'table_names', 'table_alias')):
        if len(comparison_fields) != len(field):
            raise ValueError(
                "All field lists must have the same length."
                "The length of the 'comparison_fields'"
                "does not match the length of '{}' (len = {}).".format(name, len(field))
                )

    # Check collections
    for field, name in zip((comparison_fields, groupby_fields, table_names, table_alias),
                           ('comparison_fields', 'groupby_fields', 'table_names', 'table_alias')):
        if not isinstance(field, (tuple, list)):
            raise TypeError(
                "A list-like collection must be used for database info fields."
                " A {} object was entered for '{}'.".format(type(field), name)
                )
        for value in field:
            if not isinstance(value, str):
                raise TypeError(
                    "All values in collection inputs must be of type <str>. "
                    "{} from {} is of {} type.".format(value, name, type(value))
                )

    # Confirm test_type
    test_types = ('count', 'low_distinct', 'high_distinct', 'numeric', 'id_check')
    if test_type not in test_types:
        raise ValueError(
            "The current value for 'test_type' is {} and is not a valid test type. "
            "Use a value from {}.".format(test_type, test_types)
            )
    # Confirm db_server in DB_ENG
    if db_server not in DB_ENG.keys():
        raise ValueError(
            "The value for 'db_server' is not valid. "
            "Use a value from {}.".format(DB_ENG.keys())
        )

    # Check save_location
    if save_location:
        test_input_string(save_location)
        split_test = save_location.split('\\')
        if len(split_test) > 1:
            raise ValueError(
                "Save location must use / instead of "
                "\\ to indicate sub-directories."
                )

def test_input_ut_runtest(review_threshold, test_type):
    # TODO: Add docstring
    """docstring"""
    if not isinstance(review_threshold, (int, float)):
        raise TypeError(
            "The value for 'review_threshold' must be numeric. "
            "{} is {}.".format(review_threshold, type(review_threshold))
        )

    if test_type == 'id_check':
        raise ValueError(
            "The 'run_test' method cannot be used to complete the 'id_check' test. "
            "Please use the 'compare_ids' method instead."
        )

def test_input_ut_ids(table_alias, id_fields, clear_results, remove_time):
    # TODO: Add docstring
    """docstring"""
    for field, name in zip((id_fields, table_alias), ('id_fields', 'table_alias')):
        # Only two fields for each
        if len(field) != 2:
            raise ValueError(
                "Only two fields can be compared at a time for 'compare_ids'. "
                "{} has len = {}.".format(name, len(field))
                )
        # Check for collections
        if not isinstance(field, (tuple, list)):
            raise TypeError(
                "A list-like collection must be used for database info fields."
                " A {} object was entered for {}.".format(type(field), name)
                )
        # Collect values are strings
        for value in field:
            if not isinstance(value, str):
                raise TypeError(
                    "All values in collection inputs must be of type <str>. "
                    "{} in {} is of {} type.".format(value, name, type(value))
                )

    # Test remove_results and remove_time
    for field in (clear_results, remove_time):
        if not isinstance(field, bool):
            raise TypeError(
                "Values for both 'clear_results' and 'remove_time' must be of type <bool>."
            )
def test_input_ut_summ(summary_type, save_type, remove_time, save_location):
    # TODO: Add docstring
    """docstring"""
    summary_types = ('data', 'image', 'both')
    if summary_type not in summary_types:
        raise ValueError(
            "Value for 'summary_type' ({}) must be in {}.".format(summary_type, summary_types)
        )

    save_types = ('data', 'image', 'both', False)
    if save_type not in save_types:
        raise ValueError(
            "Value for 'save_type' ({}) must be in {}.".format(save_type, save_types)
        )

    if not isinstance(remove_time, bool):
        raise TypeError(
            "Value for 'remove_time' must be of type <bool>."
        )

    if save_type:
        if not save_location:
            raise ValueError(
                "Unable to save results, the 'save_location' attribute is empty."
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
        db_server: (str) Server alias, as specified by DB_ENG.
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
        test_input_ut_init(comparison_fields=comparison_fields,
                           groupby_fields=groupby_fields,
                           table_names=table_names,
                           table_alias=table_alias,
                           db_server=db_server,
                           test_type=test_type,
                           save_location=save_location)

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

        source_str = ("SELECT {source_groupby}, {source_compare} AS "
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
        # Manage 'id_check' or custom test_str
        if self.test_type == 'id_check':
            self._create_id_check_string()
            return
        if test_string:
            test_input_string(string_var=test_string)
            self._test_str = test_string

        # Or create initial test string
        elif self.test_type == 'count':
            test_str = self._create_count_string()
        elif self.test_type == 'low_distinct':
            test_str = self._create_low_distinct_string()
        elif self.test_type == 'high_distinct':
            test_str = self._create_high_distinct_string()
        elif self.test_type == 'numeric':
            test_str = self._create_numeric_string()

        # Set up string elements
        joins = ''
        selects = ''

        # Create SELECT statement
        if self.test_type in ('count', 'high_distinct', 'numeric'):
            initial_select_state = (" SELECT {target_alias}.{target_groupby}, "
                                    "{target_alias}.row_count AS {target_alias}_count")\
                                   .format(target_alias=self.table_alias[0],
                                           target_groupby=self.groupby_fields[0])

            join_state = (" JOIN {alias} ON {alias}.{groupby_field} "
                          "= {target_alias}.{target_groupby}")
            for alias, groupby_field in zip(self.table_alias[1:], self.groupby_fields[1:]):
                joins += join_state.format(alias=alias,
                                           groupby_field=groupby_field,
                                           target_alias=self.table_alias[0],
                                           target_groupby=self.groupby_fields[0])

            order = " ORDER BY {target_alias}.{target_groupby}"\
                        .format(target_alias=self.table_alias[0],
                                target_groupby=self.groupby_fields[0])
        else:
            initial_select_state = (" SELECT {target_alias}.{target_groupby}, "
                                    "{target_alias}.{target_compare} AS"
                                    " {target_alias}_{target_compare}, "
                                    "{target_alias}.row_count AS {target_alias}_count")\
                                   .format(target_alias=self.table_alias[0],
                                           target_compare=self.comparison_fields[0],
                                           target_groupby=self.groupby_fields[0])

            join_state = (" LEFT JOIN {alias} ON {alias}.{groupby_field} = "
                          "{target_alias}.{target_groupby}"
                          " AND {alias}.{compare} = {target_alias}.{target_compare}")
            for alias, groupby_field, compare_field in zip(self.table_alias[1:],
                                                           self.groupby_fields[1:],
                                                           self.comparison_fields[1:]):
                joins += join_state.format(alias=alias,
                                           groupby_field=groupby_field,
                                           compare=compare_field,
                                           target_alias=self.table_alias[0],
                                           target_groupby=self.groupby_fields[0],
                                           target_compare=self.comparison_fields[0])

            order = (" ORDER BY {target_alias}.{target_groupby}, "
                     "{target_alias}.{target_compare}")\
                    .format(target_alias=self.table_alias[0],
                            target_groupby=self.groupby_fields[0],
                            target_compare=self.comparison_fields[0])

        table_select = ", {alias}.row_count AS {alias}_count"
        for alias in self.table_alias[1:]:
            selects += table_select.format(alias=alias)

        # Create FROM statement
        initial_from_state = " FROM {target_alias}".format(target_alias=self.table_alias[0])

        # Create test string
        test_str += initial_select_state + selects + initial_from_state \
                    + joins + order
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
        if self._results[assess_col].isnull().sum() == self._results.shape[0]:
            assessment = 'MISSING VALUE for ' + comparison_col + '_' + self.comparison_fields[0]
            print(assessment)
            return assessment

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
        test_input_ut_runtest(review_threshold=review_threshold, test_type=self.test_type)

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
                self._results[perc_col] = \
                ((self._results[target_col] - self._results[col_name])\
                /self._results[target_col] * 100).astype(float).round(2)
                # Assess perc diff
                assessment = self._assess_priority_review(col, perc_col,
                                                          review_threshold=review_threshold)
                if assessment:
                    self._priority_review[test_field + '_' + col] = assessment
                # Assign to summary
                summary_field = self.groupby_fields[0]
                if self._summary.empty:
                    if self.test_type == 'low_distinct':
                        self._summary = self._results.groupby(summary_field)[perc_col]\
                                       .mean().reset_index()
                    else:
                        self._summary = self._results[[summary_field, perc_col]].copy()
                    self._summary.rename(columns={perc_col: '{}_{} [{}]'.format(test_field,
                                                                                col,
                                                                                self.test_type)},
                                         inplace=True)
                else:
                    if self.test_type == 'low_distinct':
                        summary_col = self._results.groupby(summary_field)[perc_col]\
                                      .mean().reset_index()
                    else:
                        summary_col = self._results[[summary_field, perc_col]].copy()
                    summary_col.rename(columns={perc_col: '{}_{} [{}]'.format(test_field,
                                                                              col,
                                                                              self.test_type)},
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

    def compare_ids(self, table_alias, id_fields, clear_results=True, remove_time=True):
        """
        Complete a count comparison based on stored data and combine with a comparison of IDs.
        Saves results if save_location is stored.

        inputs:
            table_alias: (list-like) Two table alias to compare.
                         "Target" table must be listed first.
            id_fields: (list-like) Names of ID fields to compare.
                       "Target" table field must be listed first.
            clear_results: (optional, bool, default=True) If True, clear _results.
                           If False, retains results. (If moving on to other tests
                           it is recommended to clear results.)
            remove_time: (optional, bool, default=True) If True, if groupby is a
                         datetime field, then the time component is removed.
        """
        test_input_ut_ids(table_alias=table_alias, id_fields=id_fields,
                          clear_results=clear_results, remove_time=remove_time)

        # Update for test type
        self.test_type = 'id_check'
        source_position = self.table_alias.index(table_alias[1])
        self.comparison_fields = id_fields
        self.groupby_fields = (self.groupby_fields[0], self.groupby_fields[source_position])
        self.table_names = (self.table_names[0], self.table_names[source_position])
        self.table_alias = table_alias

        # Get id data
        target_df, source_df = self.gather_data()

        # Run count test
        self.test_type = 'count'
        self.run_test()

        # Assign values for easy reference
        target_groupby = self.groupby_fields[0]
        source_groupby = self.groupby_fields[1]
        target_id = self.comparison_fields[0]
        source_id = self.comparison_fields[1]

        # Confirm id fields are in their respective dfs
        for col, df in zip((target_id, source_id),
                           (target_df, source_df)):
            if col not in df.columns:
                raise ValueError(
                    "Each column name in 'id_fields' must be in the respective DataFrame. "
                    "{} was not found in the corresponding DataFrame.".format(col)
                    )

        # Convert groupby field of count results to index for easier comparisons
        if remove_time:
            if isinstance(self._results.loc[0, self.groupby_fields[0]], datetime):
                self._results[self.groupby_fields[0]] = \
                self._results[self.groupby_fields[0]].dt.date
        self._results.index = self._results[self.groupby_fields[0]]
        self._results.drop(self.groupby_fields[0], axis=1, inplace=True)

        # Compare ids
        target_in_source_name = table_alias[0] + '_missing_in_' + table_alias[1]
        source_in_target_name = table_alias[1] + '_missing_in_' + table_alias[0]
        for ind in self._results.index:
            print("Commencing ID comparison for", ind, "...")
            try:
                # Get ids from the specified groupby value
                is_ind_target = target_df[target_groupby] == ind
                is_ind_source = source_df[source_groupby] == ind
                # Note: Code is brittle and depends on Python version, current = 3.7.3 or lower
                # Check if the id values are in the other DataFrame
                target_in_source = target_df.loc[is_ind_target, target_id]\
                                   .isin(source_df.loc[is_ind_source, source_id])
                source_in_target = source_df.loc[is_ind_source, source_id]\
                                   .isin(target_df.loc[is_ind_target, target_id])
                # Collect missing ids
                missing_target_ids = \
                target_df.loc[~target_in_source & is_ind_target, target_id].values
                missing_source_ids = \
                source_df.loc[~source_in_target & is_ind_source, source_id].values
                # Store id values and counts in results
                self._results.loc[ind, target_in_source_name] = \
                ", ".join(str(missing_target_ids).strip('[] ').split())
                self._results.loc[ind, 'missing_' + table_alias[0] + '_ids'] = \
                missing_target_ids.shape[0]
                self._results.loc[ind, source_in_target_name] = \
                ", ".join(str(missing_source_ids).strip('[] ').split())
                self._results.loc[ind, 'missing_' + table_alias[1] + '_ids'] = \
                missing_source_ids.shape[0]
            except Exception as e:
                self._exceptions['missing_id_' + str(ind)] = e
                print('EXCEPTION missing_id', ind, ":", e)
            print('ID comparison for', ind, 'complete.')

        # Check save results
        if self.save_location:
            self.save_results(index=True)

        # Check to clear results
        if clear_results:
            self._results = None

    def summarize_results(self, summary_type='both', save_type='both', remove_time=True):
        # TODO: Add docstring
        """docstring"""
        test_input_ut_summ(summary_type=summary_type,
                           save_type=save_type,
                           remove_time=remove_time,
                           save_location=self.save_location)

        # Set index for summary df
        summary_field = self.groupby_fields[0]
        if remove_time:
            if isinstance(self._summary.loc[0, summary_field], datetime):
                self._summary[summary_field] = self._summary[summary_field].dt.date
        self._summary.index = self._summary[summary_field]
        self._summary.drop(summary_field, axis=1, inplace=True)
        # Drop rows that were only in "target" table
        self._summary.dropna(how='all', inplace=True)
        self._summary = self._summary.transpose()

        if save_type in ('data', 'both'):
            self._summary.to_csv(self.save_location + '/' + self._today_date \
                                 + '/summary_' + self._today_date + '.csv')

        if summary_type in ('image', 'both'):
            palette = sns.diverging_palette(30, 230, s=99, l=60, n=15)
            plt.figure(figsize=(self._summary.shape[1] * 1.5, self._summary.shape[0] * 0.55))
            ax = sns.heatmap(self._summary,
                             vmin=-30,
                             vmax=30,
                             cmap=palette,
                             center=0,
                             annot=True,
                             fmt='.3g',
                             annot_kws={'fontsize': 12},
                             linewidths=1)
            ax.xaxis.tick_top()
            ax.tick_params(axis='both', which='both', length=0, labelsize=12)
            ax.tick_params(axis='x', rotation=45)
            plt.xlabel('')
            if save_type in ('image', 'both'):
                plt.savefig(self.save_location + '/' + self._today_date \
                            + '/summary_img_' + self._today_date + '.png',
                            bbox_inches='tight')
            plt.show()

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

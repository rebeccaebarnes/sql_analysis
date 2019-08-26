# -*- coding: utf-8 -*-
# Authors: Rebecca Barnes <rebeccaebarnes@gmail.com>
# License: MIT
"""
The :mod:`sql_analysis` module gathers and assesses data from SQL databases.
"""
from collections import namedtuple
from datetime import datetime
import fnmatch
import os
from typing import Any, Mapping, NoReturn, Optional, Sequence, Tuple, TypeVar, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sql_config as sqlc
import sql_input_tests as sqlit


DB_ENG = sqlc.DB_ENG
P = TypeVar('P', bound=pd.DataFrame)
FieldList = namedtuple('FieldList', ['test_type', 'field_names'])

def sql_query(query_str: str, engine: str) -> P:
    """
    Complete database query using pandas.read_sql.

    Parameters:
        query_str: (str) SQL query string.
        engine: (str) Server type to use from options of DB_ENG.keys().

    Returns:
        Pandas DataFrame.
    """

    return pd.read_sql(query_str, DB_ENG[engine])

def extract_summ_dict(keyword_dict: dict) -> Tuple[str, bool]:
    """Extracts values from the key dictionary for .summarize_results."""
    summary_type = save_type = remove_time = None
    for key, value in keyword_dict.items():
        if key == 'summary_type':
            summary_type = value
        if key == 'save_type':
            save_type = value
        if key == 'remove_time':
            remove_time = value

    return summary_type, save_type, remove_time

def detect_field_type(table_name: str, field_name: str, engine: str,
                      low_distinct_thresh: int = 10, row_limit: int = 500) -> str:
    """"
    Determine the test type to be used on a database table.

    Parameters:
        table_name: (str) Name of the database table.
        field_name: (str) Name of the field to test.
        engine: (str) Server type to use from options of DB_ENG.keys().
        low_distinct_thresh: (optional, int, default=10) Number of distinct values
                             at which a field will be classified as low_distinct
                             test type.
    Returns:
        test_type: (str, {'numeric', 'high_distinct', 'low_distinct'})

    Raises:
        ValueError: If a valid database engine key is not used.
        TypeError: If the SQL dialet name is not 'postgresql', 'mysql', 'sqlite',
                   'oracle' or 'mssql'.
    """
    sqlit.test_in_collection(engine, list(DB_ENG.keys()), 'database engine')
    query_str = "SELECT {field_name} FROM {table_name}".format(field_name=field_name,
                                                               table_name=table_name)

    if row_limit:
        dialect_name = DB_ENG[engine].dialect.name
        if dialect_name in ('postgresql', 'mysql', 'sqlite'):
            query_str += " LIMIT {row_limit}"
        elif dialect_name == 'oracle':
            query_str += " WHERE ROWNUM <= {row_limit}"
        elif dialect_name == 'mssql':
            query_str = ("SELECT TOP {row_limit} {field_name} FROM {table_name}"
                         .format(field_name=field_name, table_name=table_name))
        else:
            raise TypeError(
                "The '{}' dialect is not currently supported for auto-dectection. "
                "Tests can be run by assigning test type with SQLTest. "
                "If you would like your dialect to be supported for auto-detection "
                "an issues flag or pull request can be made to "
                "https://github.com/rebeccaebarnes/sql_analysis.".format(dialect_name)
            )

    query_str = query_str.format(row_limit=row_limit)

    df = sql_query(query_str, engine)

    if np.issubdtype(df[field_name].dtype, np.number):
        test_type = 'numeric'
    elif df[field_name].nunique() > low_distinct_thresh:
        test_type = 'high_distinct'
    else:
        test_type = 'low_distinct'

    return test_type

def collect_field_type(table_name: str, field_names: Sequence[str], engine: str,
                       low_distinct_thresh: int = 10, row_limit: int = 500) -> namedtuple:
    """
    Determine test types for fields in a database table.

    Parameters:
        table_name: (str) Name of the database table.
        field_names: (list-like) Names of the field to test.
        engine: (str) Server type to use from options of DB_ENG.keys().
        low_distinct_thresh: (optional, int, default=10) Number of distinct values
                             at which a field will be classified as low_distinct
                             test type.
    Returns:
        numeric, high_distinct, low_distinct: (namedtuple) Each named tuple contains
                 the .test_type and .field_names that have been assigned to it.
    """
    numeric = []
    high_distinct = []
    low_distinct = []

    print('Commending field type detection...')
    for field in field_names:
        test_type = detect_field_type(table_name, field, engine, low_distinct_thresh, row_limit)
        if test_type == 'numeric':
            numeric.append(field)
        if test_type == 'high_distinct':
            high_distinct.append(field)
        if test_type == 'low_distinct':
            low_distinct.append(field)

    # Convert to named tuples
    numeric = FieldList('numeric', numeric)
    high_distinct = FieldList('high_distinct', high_distinct)
    low_distinct = FieldList('low_distinct', low_distinct)

    print('Field type detection complete.\n')
    return numeric, high_distinct, low_distinct

def detect_field_names(table_name: str, engine: str) -> Sequence[str]:
    """
    Determine field names in a table.

    Parameters:
        table_name: (str) Name of the database table.
        engine: (str) Server type to use from options of DB_ENG.keys().

    Returns:
        (list) Names of table fields.

    Raises:
        ValueError: If a valid database engine key is not used.
    """
    sqlit.test_in_collection(engine, list(DB_ENG.keys()), 'database engine')
    query_str = "SELECT * FROM {}".format(table_name)
    df = sql_query(query_str, engine)

    return df.columns.tolist()

def collect_field_names(table_names: Sequence[str], engine: str) -> Sequence[Sequence[str]]:
    """
    Collect and group corresponding field names from database tables.

    Use for 'compare_tables' requires default order of table fields to correspond
    to each other.
    """
    table_collect = []

    for table in table_names:
        fields = detect_field_names(table, engine)
        table_collect.append(fields)

    table_fields = []
    for fields in zip(*table_collect):
        table_fields.append(fields)

    return table_fields

class SQLTest:
    """
    Complete SQL queries and equality comparisons between database tables.

    Parameters:
        table_names: (list-like) Name of each database table to be queried.
                     The first table entered will be used as the "template" table.
        table_alias: (list-like) Alias string for each database table to be queried.
                     Each alias must be a single word containing no underscores.
                     Alias order should be consistent with that of 'table_names'.
        groupby_fields: (list-like) Name of field used to group for each table.
                        Field order should be consistent with that of 'table_names'.
        comparison_fields: (list-like) Name of field to be queried for each table.
                          Field order should be consistent with that of 'table_names'.
        db_server: (str) Server alias, as specified by DB_ENG.
        test_type: {'count', 'low_distinct', 'high_distinct', 'numeric', 'id_check'}.
                   See .create_test_string method for more details on test types.
        save_location: (optional, str) Folder directory for saving.

    Attributes:
        _alt_save_loc: (str) Stores alternative save location.
        _test_str: (str) Stores test string created by create_test_string method.
        _results: (pandas DataFrame) Stores DataFrame generated by run_test method.
        _summary: (pandas DataFrame) Stores DataFrame with single summary value
                  of difference between fields for each groupby value. If
                  additional 'run_test's are completed without clearing _summary,
                  the additional summary data will be concatenated.
        _exceptions: (dict) Stores exceptions encountered while completing run_test
                     and compare_ids.
        _priority_review: (dict) Stores information on fields with large differences
                          between tables.
        _today_date: (str) Stores current date in %y%m%d format.
        _alt_date: (str) Stores current date in %d-%b-%y format.

        To reduce size, all attributes other than _today_date and _alt_date
        can be cleared prior to pickling with no adverse affects.

    Raises (on init):
        ValueError: If only one table is provided,
                    if the length of the table info fields don't match,
                    if the test_type or db_server values are  not valid, or
                    if the save location uses '\\' instead of '/'.

    Methods:
        update: Update class attributes.
        clear_private_attr: Clears all private attributes except dates.
        create_test_string: Create SQL query string from specified inputs.
        gather_data: Complete database query based on specified SQL string.
        run_test: Complete the equality comparison between the '_count' columns.
        save_results: Create folder directory as needed and save results to this location.
        compare_ids: Complete a comparison of counts and id fields.
    """
    def __init__(self, table_names: Sequence[str], table_alias: Sequence[str],
                 groupby_fields: Sequence[str], comparison_fields: Sequence[str],
                 db_server: str, test_type: str, save_location: Optional[str] = None) -> NoReturn:
        sqlit.test_input_init(comparison_fields=comparison_fields,
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
        self._alt_save_loc = None
        self._test_str = None
        self._results = None
        self._summary = pd.DataFrame([])
        self._exceptions = {}
        self._priority_review = {}
        self._today_date = datetime.today().strftime('%y%m%d')
        self._alt_date = datetime.today().strftime('%d-%b-%y')

    def update(self, keyword_dict: dict) -> NoReturn:
        """
        Update class attributes with test validation.

        Raises:
            ValueError: If only one table is provided,
                        if the length of the table info fields don't match,
                        if the test_type or db_server values are  not valid, or
                        if the save location uses '\\' instead of '/'.
        """
        for key, value in keyword_dict.items():
            setattr(self, key, value)
        sqlit.test_input_init(self.table_names, self.table_alias, self.groupby_fields,
                              self.comparison_fields, self.db_server, self.test_type,
                              self.save_location)

    def clear_private_attr(self) -> NoReturn:
        """
        Clear _alt_save_loc, _test_str, _results, _summary, _exceptions,
        _priority_review attributes.
        """
        self._alt_save_loc = None
        self._test_str = None
        self._results = None
        self._summary = None
        self._exceptions = None
        self._priority_review = None

    def _create_count_string(self):
        """
        Create CTE components of a SQL query string to test counts between
        stored gropuby fields.
        """
        test_str = ""

        # Create target CTE
        cte_statement = ("WITH {target_alias} AS (SELECT {target_groupby}, COUNT(*) AS row_count "
                         "FROM {target_table} GROUP BY {target_groupby})")
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = (", {alias} AS (SELECT {groupby}, COUNT(*) AS row_count"
                     " FROM {table} GROUP BY {groupby})")
        for alias, groupby_field, table in zip(self.table_alias[1:],
                                               self.groupby_fields[1:],
                                               self.table_names[1:]):
            test_str += table_cte.format(alias=alias, groupby=groupby_field, table=table)

        return test_str

    def _create_low_distinct_string(self):
        """
        Create CTE components of a SQL query string to test values between
        fields with few distinct values from stored comparison and groupby fields.
        """
        test_str = ""

        # Create target CTE
        cte_statement = ("WITH {target_alias} AS (SELECT {target_groupby}, "
                         "COALESCE(CAST({target_compare} AS varchar(255)), 'Unknown') "
                         "AS {target_compare}, COUNT(*) AS row_count "
                         "FROM {target_table} GROUP BY {target_groupby}, "
                         "COALESCE(CAST({target_compare} AS varchar(255)), 'Unknown'))")
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_compare=self.comparison_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = (", {alias} AS (SELECT {groupby}, "
                     "COALESCE(CAST({compare} AS varchar(255)), 'Unknown') AS {compare}, "
                     "COUNT(*) AS row_count FROM {table} GROUP BY {groupby}, "
                     "COALESCE(CAST({compare} AS varchar(255)), 'Unknown'))")
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
        """
        Create CTE components of a SQL query string to test values between
        fields with many distinct values from stored comparison and groupby fields.
        """
        test_str = ""

        # Create target CTE
        cte_statement = ("WITH {target_alias} AS (SELECT {target_groupby}, "
                         "COUNT(DISTINCT {target_compare}) AS row_count"
                         " FROM {target_table} GROUP BY {target_groupby})")
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_compare=self.comparison_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = (", {alias} AS (SELECT {groupby}, COUNT(DISTINCT {compare}) "
                     "AS row_count FROM {table} GROUP BY {groupby})")
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
        """
        Create CTE components of a SQL query string to test values between
        fields that are numeric from stored comparison and groupby fields.
        """
        test_str = ""

        # Create target CTE
        cte_statement = ("WITH {target_alias} AS (SELECT {target_groupby}, "
                         "SUM({target_compare}) AS row_count "
                         "FROM {target_table} GROUP BY {target_groupby})")
        test_str = cte_statement.format(target_alias=self.table_alias[0],
                                        target_groupby=self.groupby_fields[0],
                                        target_compare=self.comparison_fields[0],
                                        target_table=self.table_names[0])

        # Create other CTEs
        table_cte = (", {alias} AS (SELECT {groupby}, SUM({compare}) AS row_count"
                     " FROM {table} GROUP BY {groupby})")
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
        """
        Create SQL query strings to obtain all values from two tables from
        stored comparison and groupby fields.
        """
        target_str = ("SELECT {target_groupby}, {target_compare} FROM {target_table}"
                      .format(target_groupby=self.groupby_fields[0],
                              target_compare=self.comparison_fields[0],
                              target_table=self.table_names[0]))

        source_str = (("SELECT {source_groupby}, {source_compare} AS "
                       "{target_compare} FROM {source_table}")
                      .format(source_groupby=self.groupby_fields[1],
                              source_compare=self.comparison_fields[1],
                              target_compare=self.comparison_fields[0],
                              source_table=self.table_names[1]))

        self._test_str = (target_str, source_str)

    def create_test_string(self, test_string: Optional[str] = None) -> str:
        """
        Create SQL query string.

        If a test_string is provided, stores this string.
        If not, creates and stores SQL string based on stored comparison,
        gropuby, table names, table alias, and test type attributes. The tests
        segment results based on the values found in the groupby fields. Five test
        tests are managed:
            count: counts the number of rows
            high_distinct: counts the number of distinct values
            low_distinct: counts row by the list of values in the field
            numeric: sums the values (Nulls are replaced with 'Unknown' for joining)
            id_check: collects all rows for the groupby and id fields

        Parameters:
            test_string: (optional, str) Custom SQL query string.
        """
        # Manage 'id_check' or custom test_str
        if self.test_type == 'id_check':
            self._create_id_check_string()
            return
        if test_string:
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
            initial_select_state = ((" SELECT {target_alias}.{target_groupby}, "
                                     "{target_alias}.row_count AS {target_alias}_count")
                                    .format(target_alias=self.table_alias[0],
                                            target_groupby=self.groupby_fields[0]))

            join_state = (" JOIN {alias} ON {alias}.{groupby_field} "
                          "= {target_alias}.{target_groupby}")
            for alias, groupby_field in zip(self.table_alias[1:], self.groupby_fields[1:]):
                joins += join_state.format(alias=alias,
                                           groupby_field=groupby_field,
                                           target_alias=self.table_alias[0],
                                           target_groupby=self.groupby_fields[0])

            order = (" ORDER BY {target_alias}.{target_groupby}"
                     .format(target_alias=self.table_alias[0],
                             target_groupby=self.groupby_fields[0]))
        else:
            initial_select_state = ((" SELECT {target_alias}.{target_groupby}, "
                                     "{target_alias}.{target_compare} AS"
                                     " {target_alias}_{target_compare}, "
                                     "{target_alias}.row_count AS {target_alias}_count")
                                    .format(target_alias=self.table_alias[0],
                                            target_compare=self.comparison_fields[0],
                                            target_groupby=self.groupby_fields[0]))

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

            order = ((" ORDER BY {target_alias}.{target_groupby}, "
                      "{target_alias}.{target_compare}")
                     .format(target_alias=self.table_alias[0],
                             target_groupby=self.groupby_fields[0],
                             target_compare=self.comparison_fields[0]))

        table_select = ", {alias}.row_count AS {alias}_count"
        for alias in self.table_alias[1:]:
            selects += table_select.format(alias=alias)

        # Create FROM statement
        initial_from_state = " FROM {target_alias}".format(target_alias=self.table_alias[0])

        # Create test string
        test_str += initial_select_state + selects + initial_from_state + joins + order
        self._test_str = test_str

    def customize_test_string(self, insert_type: str, add_string: str, table_alias: str,
                              format_values: Optional[Mapping[str, Any]] = None) -> NoReturn:
        """
        Customize CTEs in query string created by create_test_string.

        Parameters:
            type: {'from, group_by'}: If 'from' string will be added after the
                  FROM statement. If 'group_by' string will be added before the
                  GROUP BY statement.
            add_string: (str) String to insert into query string.
            table_alias: (str) Alias of CTE in which to insert the statement.
            format_values: (optional, dict-like) Values to add into string if
                           '{key}' are present in 'add_string'.

        Raises:
            ValueError: If the value for 'instert_type' or 'table_alias' is not valid.
        """
        sqlit.test_in_collection(insert_type, ['from', 'group_by'], "'insert_type'")
        sqlit.test_in_collection(table_alias, self.table_alias, 'table alias')
        cte_split = self._test_str.split('), ')

        # Find correct cte
        pos = 0
        for string in cte_split:
            if table_alias + ' AS' in string:
                cte = string
                break
            pos += 1

        # Establish search term
        insert_on = 'FROM'
        if insert_type == 'group_by':
            insert_on = 'GROUP BY'

        # Find split ind
        ind = cte.find(insert_on)
        if insert_type == 'from':
            # Find the second space after FROM
            ind += cte[ind:].find(' ') + 1
            ind += cte[ind:].find(' ')

        first_half = cte[:ind].strip()
        second_half = cte[ind:].strip()

        # Create the new CTE and swap for the old one
        if format_values:
            add_string = add_string.format(**format_values)
        cte = ' '.join([first_half, add_string.strip(), second_half])
        cte_split[pos] = cte

        self._test_str = '), '.join(cte_split)

    def gather_data(self,
                    test_string: Optional[str] = None) -> Union[P, Tuple[P]]:
        """
        Complete query of database.

        If a test_string is provided, stores this string.
        If not, creates and stores SQL string based on stored comparison_fields,
        gropuby_fields, table_names, table_alias, and test_type.

        Parameters:
            test_string: (optional, str) Custom SQL query string.

        Returns:
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

    def _assess_priority_review(self, table_alias: Sequence[str], assess_col: str,
                                review_threshold: Union[int, float]) -> str:
        """
        Assess for field with missing values or high discrepancies in values.

        Parameters:
            table_alias: (str) Alias for the table that contains the field.
            assess_col: (str) Name of the column containing value differences as a percentage.
            review_threshold: (numeric) Threshold, as a percentage, above which
                              differences in values will be flagged for priority review.
        """
        assessment = None
        if (self._results[assess_col].isnull().sum() \
            or (self._results[assess_col] == 'Unknown').sum()) == self._results.shape[0]:
            assessment = 'MISSING VALUE for ' + self.comparison_fields[0] + '_' + table_alias
            print(assessment)
            return assessment

        is_not_missing = self._results[assess_col] != 100
        not_missing_median = self._results.loc[is_not_missing, assess_col].abs().median()
        if not_missing_median > review_threshold:
            assessment = 'PRIORITY REVIEW on ' + self.comparison_fields[0] \
                          + '_' + table_alias + ': ' + str(not_missing_median)
            print(assessment)
            return assessment

    def save_results(self, index: bool = False, replace: bool = True,
                     use_alt_loc: bool = False) -> NoReturn:
        """
        Create folder directory based on stored save_location, current date, and test_field.

        Parameters:
            index: (optional, boolean, default=False) Indicates whether the saved
                   csv should contain the index.
            replace: (optional, boolean, default=True) Indicates whether an existing
                     save directory should be replaced.
            use_alt_loc: (optional, boolean, default=False) Indicates whether
                         folder_name will be derived from existing names in
                         directory.
        """
        folder_name = self.save_location.strip('/') + '/' + self._today_date
        if use_alt_loc:
            folder_name = self._alt_save_loc
        elif not replace:
            if os.path.exists(folder_name):
                file_count = len(fnmatch.filter(os.listdir(self.save_location),
                                                self._today_date + '*'))
                folder_name += '_' + str(file_count)
                self._alt_save_loc = folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_name = '/' + self.comparison_fields[0] + '.csv'
        self._results.to_csv(folder_name + file_name, index=index)

    def run_test(self, test_string: Optional[str] = None,
                 review_threshold: Union[int, float] = 2,
                 replace_save: bool = True, use_alt_loc: bool = False) -> NoReturn:
        """
        Run equality comparisons between the comparison_fields.

        Calculates absolute and percentage difference between fields.
        Manages and logs exceptions. Logs missing and priority review fields.

        Stores results in '_results' and ongoing summary in _summary.
        Saves results if save_location is stored.

        Parameters:
            test_string: (str, optional) Custom SQL query string.
            review_threshold: (numeric, optional, default=2) Threshold, as a
                              percentage, above which differences in values will
                              be flagged for priority review.
            replace_save: (optional, boolean, default=True) Indicates whether an
                          existing save directory should be replaced.
            use_alt_loc: (optional, boolean, default=False) Indicates whether
                         folder_name will be derived from existing names in
                         directory.

        Raises:
            ValueError: If 'test_type' is 'id_check'.
        """
        sqlit.test_input_runtest(self.test_type)

        print('Commencing {} test for {}...'.format(self.test_type, self.comparison_fields[0]))
        self._results = self.gather_data(test_string=test_string)
        target_col = self.table_alias[0] + '_count'
        test_field = self.comparison_fields[0]
        for alias in self.table_alias[1:]:
            try:
                col_name = alias + '_count'
                # Get difference in counts
                compare_col = self.table_alias[0] + '_minus_' + alias
                self._results[compare_col] = self._results[target_col] - self._results[col_name]

                # Get perc diff
                perc_col = 'perc_diff_' + alias
                self._results[perc_col] = \
                (((self._results[target_col] - self._results[col_name])\
                /self._results[target_col] * 100)
                 .astype(float)
                 .round(2))
                # Manage missing high_distinct fields
                if self.test_type == 'high_distinct':
                    self._results.loc[self._results[perc_col] == 100, perc_col] = np.nan
                # Assess perc diff
                assessment = self._assess_priority_review(alias, perc_col,
                                                          review_threshold=review_threshold)
                if assessment:
                    self._priority_review[test_field + '_' + alias] = assessment
                # Assign to summary
                summary_field = self.groupby_fields[0]
                if self._summary.empty:
                    if self.test_type == 'low_distinct':
                        self._summary = self._results.groupby(summary_field)[perc_col]\
                                       .median().reset_index()
                    else:
                        self._summary = self._results[[summary_field, perc_col]].copy()
                    self._summary.rename(
                        columns={perc_col: '{}_{} [{}]'.format(test_field, alias, self.test_type)},
                        inplace=True
                        )
                else:
                    if self.test_type == 'low_distinct':
                        summary_col = self._results.groupby(summary_field)[perc_col]\
                                      .median().reset_index()
                    else:
                        summary_col = self._results[[summary_field, perc_col]].copy()
                    summary_col.rename(
                        columns={perc_col: '{}_{} [{}]'.format(test_field, alias, self.test_type)},
                        inplace=True
                        )
                    self._summary = self._summary.merge(summary_col, how='outer', on=summary_field)
            except Exception as e:
                print('EXCEPTION:', e)
                self._exceptions[test_field + '_' + alias] = e

        # Add date
        if 'date' not in self._results.columns:
            self._results.insert(loc=0, column='date', value=self._alt_date)

        # Check save results
        if self.save_location:
            self.save_results(replace=replace_save, use_alt_loc=use_alt_loc)
        print('Test for {} complete.\n'.format(test_field))

    def compare_ids(self, table_alias: Sequence[str], id_fields: Sequence[str],
                    remove_time: bool = True) -> P:
        """
        Complete a count comparison based on stored attributes and combine with a comparison of IDs.

        Saves results if save_location is stored.

        Parameters:
            table_alias: (list-like) Two table alias to compare.
                         "Target" table must be listed first.
            id_fields: (list-like) Names of ID fields to compare.
                       "Target" table field must be listed first.
            remove_time: (optional, bool, default=True) If True, if groupby field is a
                         datetime field, then the time component is removed.
        Returns:
            results: (Pandas DataFrame) Shows summary of count differences and contains
                     lists of missing ids and the number of missing ids.

        Raises:
            ValueError: If more than two tables are entered for comparison.
        """
        sqlit.test_input_ids(table_alias=table_alias, id_fields=id_fields)

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
        self._results = self._results.drop(self.groupby_fields[0], axis=1)

        # Set up columns to store lists
        target_in_source_name = table_alias[0] + '_missing_in_' + table_alias[1]
        source_in_target_name = table_alias[1] + '_missing_in_' + table_alias[0]
        # Compare ids
        for ind in self._results.index:
            print("Commencing ID comparison for", ind, "...")
            try:
                # Get ids from the specified groupby value
                is_ind_target = target_df[target_groupby] == ind
                is_ind_source = source_df[source_groupby] == ind
                # Note: Code is brittle and depends on Python version, current = 3.7.3 or lower
                # Check if the id values are in the other DataFrame
                target_in_source = (target_df.loc[is_ind_target, target_id]
                                    .isin(source_df.loc[is_ind_source, source_id]))
                source_in_target = (source_df.loc[is_ind_source, source_id]\
                                    .isin(target_df.loc[is_ind_target, target_id]))
                # Collect missing ids
                missing_target_ids = \
                target_df.loc[~target_in_source & is_ind_target, target_id].values
                missing_source_ids = \
                source_df.loc[~source_in_target & is_ind_source, source_id].values
                # Store id values and counts in results
                if target_in_source_name not in self._results.columns:
                    self._results[target_in_source_name] = np.nan
                    self._results[target_in_source_name] = (self._results[target_in_source_name]
                                                            .astype(object))
                self._results.at[ind, target_in_source_name] = missing_target_ids
                self._results.at[ind, 'missing_' + table_alias[0] + '_ids'] = \
                missing_target_ids.shape[0]
                if source_in_target_name not in self._results.columns:
                    self._results[source_in_target_name] = np.nan
                    self._results[source_in_target_name] = (self._results[source_in_target_name]
                                                            .astype(object))
                self._results.at[ind, source_in_target_name] = missing_source_ids
                self._results.at[ind, 'missing_' + table_alias[1] + '_ids'] = \
                missing_source_ids.shape[0]
            except Exception as e:
                self._exceptions['missing_id_' + str(ind)] = e
                print('EXCEPTION missing_id', ind, ":", e)
            print('ID comparison for', ind, 'complete.')

        # Check save results
        if self.save_location:
            self.save_results(index=True)

        return self._results

    def summarize_results(self, summary_type: str = 'both', save_type: Union[str, bool] = 'both',
                          remove_time: bool = True, keyword_dict: Optional[dict] = None) \
                          -> Optional[P]:
        """
        Format data in _summary in DataFrame and image forms.

        Both DataFrame and image present data with fields as rows and groupby
        values as columns. The type of test that was completed is also included
        in the row name. Values contained in the cells show the median difference
        between the "target" table and all other tables tested as a percentage.

        For the image, when the "target" table values are greater than the
        "source" table values, then cells are shaded blue, when vice versa, the
        cells are shaded orange. Color intensity increases with increasing
        differences.

        Parameters:
            summary_type: (optional, {'both', 'image', 'data'}, default='both')
                          Indicates the format of the summary.
            save_type: (optional, {'both', 'data', 'image', False}, default='both')
                       Indicates what should be saved. If False, nothing is saved.
            remove_time: (optional, boolean, default=True) If the values in the
                         groupby fields are of type <datetime>, if True, converts
                         the type to <date>.
            keyword_dict: (optional, dict) Allows for passing of arguments via
                          dictionary. If keyword_dict is used, no values should
                          be provided for the other parameters.
        Returns:
            summary: (Pandas DataFrame) Returns if 'summary_type' is 'data' or 'both'.
                     Else returns None.

        Raises:
            ValueError: If the value for 'summary_type' or 'save_type' is not valid, or
                        if the values won't support correct save functionality.
            AttributeError: If the 'save_location' attribute is None but 'save_type'
                            has a valid non-False value.
        """
        # Manage keyword dict
        if keyword_dict:
            summary_type, save_type, remove_time = extract_summ_dict(keyword_dict)

        sqlit.test_input_summ(summary_type=summary_type,
                              save_type=save_type,
                              save_location=self.save_location)

        # Set index for summary df
        summary_field = self.groupby_fields[0]
        if remove_time:
            if isinstance(self._summary.loc[0, summary_field], datetime):
                self._summary[summary_field] = self._summary[summary_field].dt.date
        self._summary.index = self._summary[summary_field]
        self._summary = (self._summary.drop(summary_field, axis=1)
                                      .dropna(how='all')
                                      .sort_index()
                                      .transpose())
        # Sort names but keep count at top
        self._summary = pd.concat([self._summary.iloc[: len(self.groupby_fields) - 1].sort_index(),
                                   self._summary.iloc[len(self.groupby_fields) - 1:].sort_index()])

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
                if self._alt_save_loc:
                    save_location = self._alt_save_loc
                else:
                    save_location = self.save_location + '/' + self._today_date
                plt.savefig(save_location + '/summary_img_' + self._today_date + '.png',
                            bbox_inches='tight')
            plt.show()

        if summary_type in ('both', 'data'):
            return self._summary
        return None

U = TypeVar('U', bound=SQLTest)

def compare_tables(table_names: Sequence[str], table_alias: Sequence[str],
                   groupby_fields: Sequence[str],
                   id_fields: Sequence[str],
                   table_fields: Sequence[Sequence[str]],
                   db_server: str,
                   save_location: Optional[str] = None,
                   replace_save: bool = False,
                   use_alt_loc: bool = False,
                   review_threshold: Union[int, float] = 2,
                   low_distinct_thresh: int = 10,
                   row_limit: int = 500,
                   summ_kwargs: Optional[dict] = None) -> Tuple[P, U]:
    """
    Run data comparison tests between tables.

    Tests assume that a straight pull can be done from the tables, with no
    additional filters or joins.

    Parameters:
        table_names: (list-like) Name of each database table to be queried.
                     "Target" table should be listed first.
        table_alias: (list-like) Alias string for each database table to be queried.
                     Each alias must be a single word containing no underscores.
                     Alias order should be consistent with that of 'table_names'.
        groupby_fields: (list-like) Name of field used to group for each table.
                        Field order should be consistent with that of 'table_names'.
        id_fields: (list-like) Name of the primary id field in each table.
                   Field order should be consistent with that of 'table_names'.
        table_fields: (list-like) Groupings of field names. Each group has the
                      corresponding name of the field for each table. Field order
                      should be consistent with that of 'table_names.'
        db_server: (str) Server alias, as found in DB_ENG.keys().
        save_location: (optional, str) Folder directory for saving.
        replace_save: (optional, boolean, default=True) Indicates whether an
                      existing save directory should be replaced.
        use_alt_loc: (optional, boolean, default=False) Indicates whether
                     folder_name will be derived from existing names in
                     directory.
        review_threshold: (optional, numeric, default=2) Threshold, as a
                          percentage, above which differences in values will be
                          flagged for priority review.
        low_distinct_thresh: (optional, int, default=10) Number of distinct values
                             at which a field will be classified as low_distinct
                             test type.
        summ_kwargs: (optional, dict) Will be passed into SQLUnitTest.summarize_results.

    Returns:
        summary: (DataFrame) Returns if 'data' or 'both' is selected as the value
                             'summary_type' attribute for 'summarize_results'.
                             'both' is the default setting.
        tester: (class SQLTest) Object to provide access to exception and
                                    priority review logs.
    Raises:
        AttributeError: If 'save_location' is None but summarize_results' 'save_type'
                        has a valid non-False.
    """
    sqlit.comp_tables_input(save_location=save_location, summ_kwargs=summ_kwargs)

    tester = SQLTest(comparison_fields=id_fields,
                     groupby_fields=groupby_fields,
                     table_names=table_names,
                     table_alias=table_alias,
                     db_server=db_server,
                     test_type='count',
                     save_location=save_location)

    tester.run_test(review_threshold=review_threshold, replace_save=replace_save)

    test_fields = [fields[0] for fields in table_fields]
    numeric_fields, high_distinct_fields, low_distinct_fields = \
    collect_field_type(table_names[0], test_fields, db_server, low_distinct_thresh, row_limit)
    for test_groups in (high_distinct_fields, low_distinct_fields, numeric_fields):
        tester.test_type = test_groups.test_type
        for field in test_groups.field_names:
            try:
                tester.comparison_fields = table_fields[test_fields.index(field)]
                if not replace_save:
                    replace_save = True
                    use_alt_loc = True
                tester.run_test(review_threshold=review_threshold, replace_save=replace_save,
                                use_alt_loc=use_alt_loc)
            except Exception as e:
                print('EXCEPTION:', e)
                tester._exceptions[field] = e

    summary = tester.summarize_results(keyword_dict=summ_kwargs)

    return summary, tester

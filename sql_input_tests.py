# -*- coding: utf-8 -*-
# Authors: Rebecca Barnes <rebeccaebarnes@gmail.com>
# License: MIT
"""
`sql_input_test` tests inputs to the functions and classes of the `sql_analysis` module.
"""
from typing import Any, NoReturn, Optional, Sequence, Union
from sql_config import DB_ENG

def test_in_collection(variable: Any,
                       collection: Sequence,
                       collection_name: str) -> NoReturn:
    """# Confirm engine in DB_ENG"""
    if variable not in collection:
        raise ValueError(
            "'{}' is not a valid '{}'. "
            "Use a value from {}.".format(variable, collection_name, collection)
            )

def no_save_location() -> NoReturn:
    """Raise AttributeError"""
    raise AttributeError(
        "Unable to save results, the 'save_location' attribute is empty. "
        "Either add a location to the 'save_location' attribute or use 'save_type' "
        "to adjust the summary save options."
        )

def test_input_init(table_names: Sequence, table_alias: Sequence,
                    groupby_fields: Sequence, comparison_fields: Sequence,
                    db_server: str, test_type: str, save_location: Optional[str]) -> NoReturn:
    """Test inputs for SQLTest instantiation."""
    # Confirm minimum two fields
    if len(table_names) < 2:
        raise ValueError(
            "Minimum length for 'table_names' is 2."
            )

    # Confirm equal field length
    for field, name in zip((groupby_fields, comparison_fields, table_alias),
                           ('groupby_fields', 'table_names', 'table_alias')):
        if len(table_names) != len(field):
            raise ValueError(
                "All field lists must have the same length. The length of 'table_names' "
                "does not match the length of '{}' (len = {}).".format(name, len(field))
                )

    # Confirm test_type and db_server
    test_types = ('count', 'low_distinct', 'high_distinct', 'numeric', 'id_check')
    test_in_collection(test_type, test_types, 'test_type')
    test_in_collection(db_server, list(DB_ENG.keys()), 'database engine')

    # Check save_location
    if save_location:
        split_test = save_location.split('\\')
        if len(split_test) > 1:
            raise ValueError(
                "Save location must use / instead of \\ to indicate sub-directories."
                )

def test_input_runtest(test_type: str) -> NoReturn:
    """Test inputs for SQLTest run_test method."""
    if test_type == 'id_check':
        raise ValueError(
            "The 'run_test' method cannot be used to complete the 'id_check' test. "
            "Please use the 'compare_ids' method instead."
        )

def test_input_ids(table_alias: Sequence, id_fields: Sequence) -> NoReturn:
    """Test inputs for SQLTest compare_ids method."""
    for field, name in zip((id_fields, table_alias), ('id_fields', 'table_alias')):
        # Only two fields for each
        if len(field) != 2:
            raise ValueError(
                "Only two fields can be compared at a time when using 'compare_ids'. "
                "{} contains {} values.".format(name, len(field))
                )

def test_input_summ(summary_type: str, save_type: Union[str, bool],
                    save_location: Optional[str]) -> NoReturn:
    """Test inputs for SQLTest summarize_results method."""
    # Test collections
    summary_types = ('data', 'image', 'both')
    test_in_collection(summary_type, summary_types, 'summary_type')
    save_types = ('data', 'image', 'both', False)
    test_in_collection(save_type, save_types, 'save_type')

    if summary_types != save_types \
    and ((summary_type == 'image' and save_type in ('both', 'data')) \
         or ((summary_type == 'data') and save_type in ('both', 'image'))):
        raise ValueError(
            "Because the value for 'save_type' is currently '{save_type}' the value "
            "for 'summary_type' must be either 'both' or '{save_type}'."\
            .format(save_type=save_type)
            )

    if save_type:
        if not save_location:
            no_save_location()

def comp_tables_input(save_location: Optional[str],
                      summ_kwargs: Optional[dict]) -> NoReturn:
    """Test inputs for compare_tables."""
    # Test save location has been included if required
    if not summ_kwargs and not save_location:
        no_save_location()
    if summ_kwargs:
        for key, value in summ_kwargs.items():
            if key == 'save_type' and value and not save_location:
                no_save_location()

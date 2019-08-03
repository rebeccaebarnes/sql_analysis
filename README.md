# SQL Analysis

The SQL Analysis module assists in testing of data between SQL database tables. This is the development version of the module.

Use examples include comparing results in a view table to those derived from a star schema, or comparing results from a table derived from an external source to a table built via ETL.

## Main Features
The module contains two primary classes SQLGatherData and SQLUnitTest.

**SQLGatherData** allows generation of SQL query strings and gathering of data to complete one of five different tests `count`, `low_distinct`, `high_distinct`, `numeric`, and `id_check`. It also allows for use of custom query strings.

**SQLUnitTest** completes the five tests referenced above, by calculating actual differences and percentage differences between table values. Differences can be flagged for "priority review" to indicate fields that show large differences in values. A summary of results can also be collected and displayed.

Basic database queries can also be completed via the use of **sql_query**.

## Setup
The files `sql_secrets_example.py` and `sql_config_example.py` provide examples of how the SQLAlchemy engines can be configured. These files should be customized for personal use, and the variable `DB_ENG` in `sql_analysis.py` modified to retrieve the engines.

## Dependencies
This module utilizes:
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

## Acknowledgements
Testing is completed using the [PostgreSQL DVD Rental](http://www.postgresqltutorial.com/postgresql-sample-database/) sample database.

# SQL Analysis

The SQL Analysis module assists in testing of data between SQL database tables. This is the development version of the module.

Use examples include comparing results in a view table to those derived from a star schema, or comparing results from a table derived from an external source to a table built via ETL.

## Main Features
The module contains two primary classes SQLGatherData and SQLUnitTest.

**SQLGatherData** allows generation of SQL query strings and gathering of data to complete one of five different tests `count`, `low_distinct`, `high_distinct`, `numeric`, and `id_check`. It also allows for use of custom query strings.

**SQLUnitTest** completes the five tests referenced above, by calculating actual differences and percentage differences between table values. Differences can be flagged for "priority review" to indicate fields that show large differences in values. A summary of results can also be collected and displayed.

Basic database queries can also be completed via the use of **sql_query**.

## Functionality Overview
The concept behind the testing is that database information can often be segmented by a field, such as dates. Testing can be done by comparing field values across these groupings.

If we had a table `rental_view` and another `alt_rental_view`, we may wish to compare them to see if `alt_rental_view` is accurately capturing the information in `rental_view`.

We could view each table:

**rental_view**
![rental view][img/rental_view.PNG]

**alt_rental_view**
![alt rental view][img/alt_rental_view.PNG]

In this case, the field that can be used to group the data is `ss_dt` for `rental_view` and `ss_date` for `alt_rental_view`.

Using `SQLGatherData` we can specify the field and table information, generate the SQL and get the counts for the tables. (Note: Comparisons are not limited to only two tables, additional comparison tables can be added in the tuples)

![count data][img/count_gather.PNG]

Using `SQLUnitTest` we can complete the comparison between the fields, with a print out that flags field differences that are above a certain threshold.

![count test][img/count_test.PNG]

## Setup
The files `sql_secrets_example.py` and `sql_config_example.py` provide examples of how the SQLAlchemy engines can be configured. These files should be customized for personal use, and the variable `DB_ENG` in `sql_analysis.py` modified to retrieve the engines.

## Dependencies
This module utilizes:
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

## Acknowledgements
Testing is completed using the [PostgreSQL DVD Rental](http://www.postgresqltutorial.com/postgresql-sample-database/) sample database.

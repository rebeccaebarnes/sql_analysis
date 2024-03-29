# SQL Test

The SQL Test module assists in testing of data between SQL database tables. This is the development version of the module.

Use examples include comparing data in a view to those in a table derived from a star schema, or comparing results from a table derived from an external source to a table built via ETL.

## Installation
You can install sql_test directly from [pypi](https://pypi.org/project/sql-test/) using `pip install sql-test`.

Once installed, the files `sql_secrets.py` and `sql_config.py` provide examples of how the SQLAlchemy engines can be configured. These files should be customized for personal use. When customizing the files **do not** rename them or move them. Doing so will cause errors with any future updates.

## Main Features

- **Class: SQLTest**
    1. Creates and runs SQL database queries based on attributes provided with class instantiation or custom SQL query string.
    2. Completes five built in tests based on field-type categorizations of `count`, `low_distinct`, `high_distinct`, `numeric`, `id_check`.
    3. Flags fields above a specified difference threshold for "priority review".
    4. Displays a summary of results.
    5. Saves results and summary as specified.

- **Function: compare_tables**
    1. Auto-detects the type of test to be run.
    2. Utilizes methods of SQLUnitTest to complete a full comparison of table values.

- **Function: sql_query**
    1. Conducts basic database queries


## Functionality Overview
The concept behind the testing is that database information can often be segmented by a field, such as dates. Testing can be done by comparing field values across these groupings.

If we had three tables `rental_view`, `alt_rental_view`, and `alt_sim_rental_view` we may wish to compare them to see if the "alt" tables are accurately capturing the information in `rental_view`.

### Basic Query
We could view the tables:

**rental_view**

<p align="left">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/rental_view.PNG">
</p>

**alt_rental_view**

<p align="left">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/alt_rental_view.PNG">
</p>

**alt_sim_rental_view**

<p align="left">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/alt_sim_rental_view.PNG">
</p>

### Run Test Battery
And run tests on them:

To complete the test battery we need to determine the fields that can segment the data. In this case, the field that can be used to group the data is `ss_dt` for `rental_view` and `alt_sim_rental_view`, and `ss_date` for `alt_rental_view`.

Table information is specified on instantiation and `compare_tables` will create all query strings, gather the data, and run the tests based on the information provided. (Methods are available via SQLUnitTest to complete each step separately)

**Setup Code**
<p align="center">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/compare_tables_code.PNG">
</p>

Activity, exceptions, and fields flagged for priority review are logged during operation.

**Log Print-out**
<p align="center">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/log.PNG">
</p>

**Log Storage**
<p align="center">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/log_storage.PNG">
</p>

A summarized version of results (as a DataFrame or image), indicating the percentage difference between table fields (and the test type used), is also available via the test battery.

**Visual Summary**
<p align="center">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/results.png">
</p>

### Confirm Shared IDs
As further confirmation that are tables are replicas of each other, we may also test whether the IDs in one table are the same as the IDs in the other table.

We can use the .compare_ids method to compare between two tables at a time.

**Results**
<p align="center">
  <img src="https://raw.githubusercontent.com/rebeccaebarnes/sql_analysis/master/img/compare_ids.PNG">
</p>


## Dependencies
This module utilizes:
- Python v 3.5+
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

## Acknowledgements
Testing is completed using the [PostgreSQL DVD Rental](http://www.postgresqltutorial.com/postgresql-sample-database/) sample database.

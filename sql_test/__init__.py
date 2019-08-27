import warnings
from .sql_test import DB_ENG, sql_query, detect_field_type, collect_field_type, \
detect_field_names, collect_field_names, SQLTest, compare_tables

if 'dvd' in DB_ENG.keys() and 'pp' in DB_ENG.keys():
    warnings.warn("The current configurations for sql_config.py and sql_secrets.py"
                  " use example settings. To utilize the 'sql_test' module you may"
                  " need to configure these to your personal settings. These files"
                  " are found in your Python install location under in the"
                  " Lib/site-packages/sql_test/ directory.")

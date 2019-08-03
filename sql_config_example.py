"""Configure SQL server engines."""
from sqlalchemy import create_engine
import sql_secrets_example as sqls

PP_STR = 'postgresql+psycopg2://postgres:' + sqls.PASSWORD + '@localhost:5433/parch_and_posey'
ENGINE_PP = create_engine(PP_STR)

DVD_STR = 'postgresql+psycopg2://postgres:' + sqls.PASSWORD + '@localhost:5433/dvdrental'
ENGINE_DVD = create_engine(DVD_STR)

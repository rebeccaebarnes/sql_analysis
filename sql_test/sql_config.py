"""Configure SQL server engines."""
from sqlalchemy import create_engine
from .sql_secrets import PASSWORD

PP_STR = 'postgresql+psycopg2://postgres:' + PASSWORD + '@localhost:5433/parch_and_posey'
ENGINE_PP = create_engine(PP_STR)

DVD_STR = 'postgresql+psycopg2://postgres:' + PASSWORD + '@localhost:5433/dvdrental'
ENGINE_DVD = create_engine(DVD_STR)

DB_ENG = {'dvd': ENGINE_DVD,
          'pp': ENGINE_PP}

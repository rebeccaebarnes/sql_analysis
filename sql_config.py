from sqlalchemy import create_engine
import cx_Oracle
import sql_secrets as sqls

# Set up dev connection to db
ENGINE_STR_DEV = 'oracle+cx_oracle://{user}:{password}@(DESCRIPTION= \
(ADDRESS= (PROTOCOL={protocol})(HOST={host})(PORT={port}))(CONNECT_DATA= \
(SERVER={server})(SERVICE_NAME={service_name})))'\
.format(user=sqls.USER, password=sqls.PASSWORD, protocol=sqls.PROTOCOL,
        host=sqls.HOST_DEV, port=sqls.PORT, server=sqls.SERVER,
        service_name=sqls.SERVICE_NAME_DEV)
ENGINE_DEV = create_engine(ENGINE_STR_DEV)

# Set up prod connection to db
ENGINE_STR_PROD = 'oracle+cx_oracle://{user}:{password}@(DESCRIPTION= \
(ADDRESS= (PROTOCOL={protocol})(HOST={host})(PORT={port}))(CONNECT_DATA= \
(SERVER={server})(SERVICE_NAME={service_name})))' \
.format(user=sqls.USER, password=sqls.PASSWORD, protocol=sqls.PROTOCOL,
        host=sqls.HOST_PROD, port=sqls.PORT, server=sqls.SERVER,
        service_name=sqls.SERVICE_NAME_PROD)
ENGINE_PROD = create_engine(ENGINE_STR_PROD)

# Set up ss connection to db
ENGINE_STR_SS = 'oracle+cx_oracle://{user}:{password}@(DESCRIPTION= \
(ADDRESS= (PROTOCOL={protocol})(HOST={host})(PORT={port}))(CONNECT_DATA= \
(SERVER={server})(SERVICE_NAME={service_name})))' \
.format(user=sqls.USER, password=sqls.PASSWORD, protocol=sqls.PROTOCOL,
        host=sqls.HOST_SS, port=sqls.PORT, server=sqls.SERVER,
        service_name=sqls.SERVICE_NAME_SS)
ENGINE_SS = create_engine(ENGINE_STR_SS)

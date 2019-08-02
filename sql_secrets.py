'''Oracle access info'''
import getpass

HOST_DEV = 'wfdbdev.bchousing.org'
HOST_PROD = 'wfdb.bchousing.org'
HOST_SS = 'wfdb.bchousing.org'
PORT = '1521'
PROTOCOL = 'TCP'
SERVER = 'DEDICATED'
SERVICE_NAME_DEV = 'webfocusdev.bchousing.org'
SERVICE_NAME_PROD = 'webfocus.bchousing.org'
SERVICE_NAME_SS = 'dms.bchousing.org'
USER = input('Enter username: ')
PASSWORD = getpass.getpass('Enter password: ')

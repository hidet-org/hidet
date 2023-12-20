import os
import mysql.connector

def get_db_conn():
    conn = mysql.connector.connect(
        host=os.environ.get('CI_DB_HOSTNAME'),
        user=os.environ.get('CI_DB_USERNAME'),
        password=os.environ.get('CI_DB_PASSWORD'),
        port=os.environ.get('CI_DB_PORT'),
        database='hidet_ci'
    )
    return conn

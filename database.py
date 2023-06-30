import psycopg2
from psycopg2 import Error

class Database:
    @staticmethod
    def create_connection():
        connection = psycopg2.connect(user="postgres",
                                      password="postgres",
                                      host="0.0.0.0",
                                      port="5432",
                                      database="postgres")
        return connection

    @staticmethod
    def create_table():
        connection = Database.create_connection()

        with open("table.sql", "r") as f:
            sql = f.read()

        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

        cursor.close()
        connection.close()

    @staticmethod
    def insert_featuremap(featuremap, path_image):
        connection = Database.create_connection()

        cursor = connection.cursor()
        cursor.execute(f"INSERT INTO images VALUES ({featuremap}, '{path_image}');")
        connection.commit()

        cursor.close()
        connection.close()

    @staticmethod
    def select_all():
        connection = Database.create_connection()

        cursor = connection.cursor()
        cursor.execute(f"select * from images;")

        records = cursor.fetchall()

        cursor.close()
        connection.close()

        return records

# try:
#     connection = psycopg2.connect(user="postgres",
#                                   password="postgres",
#                                   host="0.0.0.0",
#                                   port="5432",
#                                   database="postgres")
#
#     cursor = connection.cursor()
#     cursor.execute("SELECT version();")
#     cursor.commit()
#
#
# except (Exception, Error) as error:
#     print("Ошибка при работе с PostgreSQL", error)
# finally:
#     cursor.close()
#     connection.close()
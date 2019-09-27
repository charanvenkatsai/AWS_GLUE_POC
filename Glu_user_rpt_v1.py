%pyspark
import requests
import json
import boto3
import datetime
from awsglue.transforms import *
from awsglue.dynamicframe import DynamicFrame
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window


INFLUX_URL = "http://influxdb-qa.edcastcloud.net:8086/query"
TABLE1 = "test5.autogen.cards"
TABLE2 = "test5.autogen.user_scores"
TABLE3 = "test5.autogen.\"users\""
S3_BUCKET_FOR_BOOKMARK = "edcast-qa-paperclip-store"
BOOKMARK_FILE = "aws-glue-nclouds/data/last_run_time"
MYSQL_DB = "poc"
MYSQL_USERS_TABLE = "edcast_qa_ecl_users"
MYSQL_CARDS_TABLE = "edcast_qa_ecl_cards"
OUTPUT_FILE_LOC = "s3://edcast-qa-paperclip-store/aws-glue-nclouds/data/influx"


def get_influx_dataframe(table, influx_url=INFLUX_URL,
                         s3_bucket_for_bookmark=S3_BUCKET_FOR_BOOKMARK,
                         bookmark_file=BOOKMARK_FILE):
    """
    Reads table from URL of influx and returns it as PySpark DataFrame

    Arguments
    ---------
        table (str):
            Table to be retrieved from Influx Database
        influx_url (str):
            URL of influx database
        s3_bucket_for_bookmark (str):
            Name of s3 bucket containing bookmark
        bookmark_file (str):
            Location of file in s3 bucket containing bookmark

    Returns
    -------
        influx_df (pyspark.sql.DataFrame):
            Influx database retrieved as pyspark.sql.DataFrame where time
            is datetime-encoded
    """
    s3 = boto3.resource('s3')
    obj = s3.Object(s3_bucket_for_bookmark, bookmark_file)
    insert_into_timestamp = obj.get()['Body'].read().decode('utf-8')
    print("Inserted into timestamp: " + insert_into_timestamp)

    # Converting insert_into_timestamp into datetime format.
    # NOTE: Python's `datetime` module supports only until microseconds. So,
    # insert_into_timestamp has been trimmed from nanoseconds to microseconds
    # for parsing into datetime.strptime()
    temp = insert_into_timestamp[:-4]
    datetime_insert_timestamp = datetime.datetime.strptime(temp, '%Y-%m-%dT%H:%M:%S.%f')

    glueContext = GlueContext(SparkContext.getOrCreate())

    if table == TABLE1:
        query = f"SELECT * FROM {table} WHERE event =~ /card_viewed|card_created|card_marked_as_complete|channel_followed|group_user_added|card_assigned/ and time > now() - 3d and time < now() - 1d"
    elif table == TABLE2:
        query = f"SELECT * FROM {table} WHERE event =~ /card_viewed|card_created|card_marked_as_complete|channel_followed|group_user_added|card_assigned/ and time > now() - 3d and time < now() - 1d"
    elif table == TABLE3:
        query = f"SELECT * FROM {table} WHERE time > now() - 3d and time < now() - 1d"

    params = {"pretty": "false", "q": query}        


    #params = {"pretty": "false", "q": "SELECT * FROM "+influxTable+" WHERE time >'"+last_processed_timestamp+"' order by time desc"}

    r = requests.get(influx_url, params=params)
    data = json.loads(r.text)

    # Retreiving data column names and values from JSON
    values = data["results"][0]["series"][0]["values"]
    columns = data["results"][0]["series"][0]["columns"]

    # Data is inserted after timestamp of first value row
    inserted_into_timestamp = values[0][0]

    # Defining schema for the data
    column_structfields = [StructField(column, StringType(), True) for column in columns]
    schema = StructType(column_structfields)

    # Creating new dataframe with above-defined schema
    df = glueContext.createDataFrame(values, schema)

    # Typecasting 'time' in 'df' to 'timestamp' format
    new_df = df.withColumn("time", df["time"].cast("timestamp"))
    new_df = new_df.withColumn("time_string",
                               date_format(new_df.time, "yyyy-MM-dd hh:mm:ss"))

    return new_df


def parse_user_cards(cards_table=TABLE1,
                     user_scores_table=TABLE2,
                     users_table = TABLE3,
                     db=MYSQL_DB,
                     db_users_table=MYSQL_USERS_TABLE,
                     db_cards_table=MYSQL_CARDS_TABLE):
    """
    Description

    Arguments:
    ----------
        cards_table (str):
            "cards" table in the Influx Database
        user_scores_table (str):
            "user_scores" table in the Influx Database
        user_table (str):
            "users" table in the Influx Database
        db (str):
            Name of MySQL Database
        db_users_table (str):
            Name of "users" table in MySQL Database
        db_cards_table(str):
            Name of Cards table in Mysql Database

    Returns:
    --------
        (str):
    """
    glueContext = GlueContext(SparkContext.getOrCreate())

    # Getting data from "user_scores" table and "cards" table
    influx_user_scores_df = get_influx_dataframe(table=user_scores_table)
    influx_cards_df = get_influx_dataframe(table=cards_table)
    influx_users_df = get_influx_dataframe(table=users_table)

    # Dropping duplicates from influx_users_df and influx_cards_df
    influx_cards_df = influx_cards_df.drop_duplicates(['org_id', 'user_id', 'card_id', 'assigned_to_user_id'])
    influx_users_df = influx_users_df.drop_duplicates(["org_id", "user_id", "created_user_id", "follower_id",
                                                       "followed_user_id", "time", "event_time"])    
    # Ranking based on "org_id"
    influx_users_df = influx_users_df.withColumn("rank_org_id", rank().over(Window.partitionBy("org_id")\
                                                                 .orderBy(asc("time"))))\
                                                                 .where(col("rank_org_id") >= 1)
#                                                                .select("rank_org_id", "org_id", "actor_id") 

    # Creating a DynamicFrame for "users" and "cards" from the schema of MySQL Database
    cards = glueContext.create_dynamic_frame_from_catalog(database=db, table_name=db_cards_table)
    users = glueContext.create_dynamic_frame_from_catalog(database=db, table_name=db_users_table)

    # Converting "users" and "cards" to PySpark DataFrame
    users_df = users.toDF()\
                    .drop_duplicates(["organization_id", "id", "created_at"])

    # Duplicates with the following keys for the following actions are dropped:
    # 1. card_created    2. card_deleted card_assigned    3. card_assigned_dismissed
    # 4. card_assigned_deleted    5. card_dismissed    6. card_marked_as_complete
    # 7. card_marked_as_uncomplete    8. card_viewed
    cards_df = cards.toDF()\
                    .drop_duplicates(['organization_id', 'author_id', 'id'])

    # Selecting given columns from "users_df" from total of 62 columns and "cards_df" from
    # total of 55 columns
    users_df = users_df.select("id", "first_name", "last_name", "email",
                               "created_at", "sign_in_count", "is_suspended", "is_active")

    cards_df = cards_df.select("id", "card_type","title")

    # Concatenating "first_name" and "last_name" to create new column "full_name" in users_df
    users_df = users_df.withColumn("user_full_name",
                                   concat_ws(" ", users_df.first_name, users_df.last_name))

    # Adding "user_account_status" based on values for mutually
    # exclusive columns "is_suspended" and "is_active"
    users_df = users_df.withColumn("user_account_status", when(users_df.is_suspended == "true", "suspended")\
                                                          .when(users_df.is_active == "true", "active"))

    # Joining users from MYSQLDB and Influx on user_id
    joined_users_df = users_df.join(influx_users_df, users_df.id == influx_users_df._user_id, 'inner')\
                              .drop(influx_users_df._user_id).drop(influx_users_df.org_id)
    # Joining cards from MYSQLDB and Influx on card_id
    joined_cards_df = cards_df.join(influx_cards_df, cards_df.id == influx_cards_df._card_id, 'inner')\
                              .drop(influx_cards_df._card_id).drop(influx_cards_df.card_type)\
                              .drop(influx_cards_df.ecl_id).drop(influx_cards_df.ecl_source_name)\
                              .drop(influx_cards_df.is_public).drop(influx_cards_df.readable_card_type)\
                              .drop(influx_cards_df._user_id)
    print("Joining \'users\' table and \'cards\' tables with their corresponding influx tables completed!")

    # The columns in these df's will change based on further activity measures
    processed_users_df = joined_users_df.select("user_id", "user_full_name", "rank_org_id")
    #processed_users_df = joined_users_df.select("user_id", "user_full_name")

    processed_cards_df = joined_cards_df.select("org_id", "card_title","card_type", "user_id",
                                                "card_id", "event", "event_time","time")
    # Joined users with cards (each of which have already been joined with their corresponding influx tables)
    joined_users_cards_df = processed_users_df.join(processed_cards_df, processed_users_df.user_id == processed_cards_df.user_id, 'inner')\
                                              .drop(processed_users_df.user_id)
                                              

    joined_users_cards_df.show(1000)

parse_user_cards()

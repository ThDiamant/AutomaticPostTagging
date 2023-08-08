import pymongo as mongo
import csv


def check_create_collection(mongoDb, collection):
    """
    Checks if collection exists in mongoDb, and if it doesn't it creates it.
    """
    if collection in mongoDb.list_collection_names():
        print(f"Collection {collection} already exists, proceeding.")
    else:
        mongoDb.create_collection(collection)
        print(f"Collection {collection} created.")

    return mongoDb[collection]

def check_db_connection(client):
    """
    Check if the connection to the db was successful
    """
    try:
        db = client.admin
        server_info = db.command('serverStatus')
        print('Connection to MongoDB server successful.')
    except mongo.errors.ConnectionFailure as e:
        print('Connection to MongoDB server failed: %s' % e)

def create_document(row):
    """
    Inputs:
        > row <list of str>: List of strings containing the data of each column.

    Creates a document to be inserted into MongoDB.
    """

    myDocument = {}
    myDocument['id'] = row[0]
    myDocument['title'] = row[1]
    myDocument['body'] = row[2]
    myDocument['tags'] = row[3]

    return myDocument

def save_document(myCollection, myDocument):
    """
    Inputs:
        > myCollection: MongoDB collection in which we want to add myDocument.
        > myDocument <dict>: Document to be inserted.
    Adds myDocument in myCollection and checks if it was inserted successfully. If
    If myDocument already exists in myCollection, if it cannot find it, a ValueError
    is raised.
    """

    try:
        # Insert myDocument in Mongo
        result = myCollection.insert_one(myDocument)
        # Check if the document was inserted successfully
        if not result.inserted_id:
            raise Exception(f"Document {myDocument} not inserted.")
        else:
            print('Document saved.')
    # If record already exists
    except mongo.errors.DuplicateKeyError:
        # Try to find the document in the DB
        query = {
            'type': myDocument['type'],
            'data.dateTime': myDocument['data']['dateTime']
        }

        existing_doc = myCollection.find_one(query)
        if existing_doc is None:
            # Something went wrong, raise an error
            raise ValueError("Cannot find existing document in the collection.")
        else:
            # Document already exists, ignore it
            pass
    except Exception as e:
        print('Error: %s' % e)

def create_and_save_document(myCollection, row):
    """
    Creates and saves document into fitbitCollection
    """
    dataDocument = create_document(row)
    print('Document created.')
    save_document(myCollection, dataDocument)

# Globals
DB_NAME = "StackOverflow"
DATA_COLLECTION_NAME = "StackOverflowCollection"
NUM_POINTS = 1000

client = mongo.MongoClient('localhost', 27017)

# Check if connection to Mongo is successful (if not, an exception will be thrown)
check_db_connection(client)
# Drop db if you want to
# client.drop_database(DB_NAME)

# Connect to the db and the collection where the data are stored or create them if they don't exist
db = client[DB_NAME]
collection = check_create_collection(db, DATA_COLLECTION_NAME)
print(db.list_collection_names())

input_file = 'training_data.tsv'

with open(input_file, 'r', newline='') as file:
    reader = csv.reader(file, delimiter='\t')
    data_points = 0
    for i, row in enumerate(reader):
        if i == 0:  # Skip the header row
            continue

        # create_and_save_document(collection, row)
        data_points += 1
        if data_points == NUM_POINTS:
            break







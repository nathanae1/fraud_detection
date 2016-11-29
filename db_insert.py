# THIS SCRIPT IS USED TO INSERT A FILE INTO A DATABASE
from pymongo import MongoClient

def write_one_to_database(database, table_name, entry, prediction):
    '''
    INPUT: string database, string table_name, dict entry, string prediction
    OUTPUT: NONE
    Accepts strings for the name, and table_name of a database
    that you want to insert a row into and inserts the entry
    into it.
    USE: write_one_to_database('fraud_dump', 'events', {'test':'test'})
    '''
    client = MongoClient()
    db = client[database]
    tab = db[table_name]
    entry['prediction'] = prediction
    tab.insert(entry)

def predict(model_location, preprocessor_location, entry):
    import cPickle as pickle
    '''
    INPUT: string model_location, string preprocessor_location, dict entry
    OUTPUT: prediction
    Accepts location of pickled model and preprocessor and uses them to make a
    prediction on the input data.
    USE: returned_prediction = predict('test_model', 'test_preprocessor', entry)
    '''
    model = None
    preprocessor = None
    with open(model_location) as model_file:
        model = pickle.load(model_file)
    with open(preprocessor_location) as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)

    return model.predict(preprocessor(entry))

# if __name__ == '__main__':
#     write_one_to_database('fraud_dump', 'events', {'test':'test'})

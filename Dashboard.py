from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import UpdateMany, UpdateOne
import json



client = MongoClient()
db = client['fraud']
collection = db['cases']


json_file = collection.find()[0]
df = pd.DataFrame(list(json_file)).to_html()
# data = pd.read_json()
# dashboard_Stuff = data[['org_name','object_id','event_created','acct_type']]


def queryDB():
    '''
    Querries database of newly created fraud cases
    return: pandas dataframe
    '''
    query = collection.find({"Fraud": True})
    data = pd.DataFrame(entry for entry in query.next())[['_id','time_recieved','org_name','object_id','Fraud probability']]
    data.sort(columns='time_recieved', ascending=False, inplace=True)
    return data.to_html()

def updateDB(docs):
    '''
    takes user input and updates the cases in the database
    '''

    requests = []
    for doc in docs:
        requests.append(UpdateOne({'object_id': doc['object_id']}))
    result = collection.bulk_write(requests)

    pass

def createDashboard(dataframe):
    '''
    INPUT: pandas dataframe
    OUTPUT: stuff ot pass to html code that creates table
    '''


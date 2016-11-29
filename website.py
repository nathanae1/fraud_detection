from flask import Flask, url_for, request, render_template
from collections import Counter
from pymongo import MongoClient
import cPickle as pickle
import pandas as pd
import json
import requests
import socket
import time
from preprocessing import preprocess

app = Flask(__name__)

# -----------------------------------------------------------------------------#
# Random stuff that I'm not passing around:
# Used to process the input data and pass it to the database
client = MongoClient()
db = client['fraud_dump']
tab = db['events']
model_location = 'model.pkl'
with open(model_location) as model_file:
        model = pickle.load(model_file)

# Used to connect to the server that gives input data
my_port = 5000
register_url = "http://10.3.35.189:5000/register"

full_table = None
# -----------------------------------------------------------------------------#


@app.route('/hello')
def api_hello():
    return 'Hello, World!'


@app.route('/')
def api_dashboard():
    json_file = tab.find()
    df = pd.DataFrame(list(json_file))[['prediction', 'org_name', 'name',
                                        'country']].to_html()
    return render_template('table.html', table=df)


@app.route('/score', methods=['POST'])
def hidden_route():
    '''
    Used to recieve calls from the server giving new data.
    '''
    # Recieving the raw data:
    #       string json output
    text = json.dumps(request.json,
                      sort_keys=True,
                      indent=4,
                      separators=(',', ': '))
    print 'hello'
    # with open('data/example.json') as f:
    #     json.dump(f)

    # print text

    # Cleaning the data:
    #       string json input -> pandas dataframe output
    input_data = json.loads(text)
    clean_data = preprocess(input_data)

    # Predicting on the data:
    #       pandas dataframe input -> boolean output
    prediction = model.predict_proba(clean_data)

    # Appending the prediction and time recieved to the cleaned data:
    #       pandas dataframe input -> dict output
    # return_data = clean_data.to_dict(orient='list')

    input_data['prediction'] = str(prediction[0][0])

    # return_data = dict((k, str(v[0])) for k, v in return_data.iteritems())

    input_data['time_received'] = time.time()

    # Writing the full data to the database:
    #       dict input

    tab.insert(input_data)

    return ''
# @app.route('/score', methods=['POST', 'GET'])
# def test_score():
#     text = json.dumps(request.json, sort_keys=True, indent=4, separators=(','
#        , ': '))
#     out_dict = json.loads(text)
#     out_dict['time_recieved'] = time.time()
#     tab.insert(out_dict)
#     df = pd.DataFrame(pd.Series(out_dict)).T
#     print df
#     return ''


@app.route('/test', methods=['GET'])
def test():
    all_data = tab.find()
    return pd.DataFrame([all_data.next() for _ in xrange(10)]).to_html()


def register_for_ping():
    my_ip = socket.gethostbyname(socket.gethostname())
    print "attempting to register %s:%d" % (my_ip, my_port)
    registration_data = {'ip': my_ip, 'port': my_port}
    requests.post(register_url, data=registration_data)


if __name__ == '__main__':
    register_for_ping()

    app.run(host='0.0.0.0', port=5000, debug=True)

'''
------------------------------Code to cannibalize------------------------------
'''
# def dict_to_html(d):
#     return '<br>'.join('{0}: {1}'.format(k, d[k]) for k in sorted(d))
#
# @app.route('/')
# def api_root():
#     return '''Welcome
#         <form action="/word_counter" method='POST' size=300>
#             <input type="textfield" name="user_input" size=300/>
#             <input type="submit" />
#         </form>
#         '''
#
# @app.route('/word_counter', methods=['POST'] )
# def api_word_counter():
#     text = str(request.form['user_input'])
#     word_counts = Counter(text.lower().split())
#     page = 'There are {0} words.<br><br>Individual word counts:<br> {1}'
#     return page.format(len(word_counts), dict_to_html(word_counts))
'''
--------------------------------------------------------------------------------
'''

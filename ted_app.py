import flask
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

path = '/Volumes/ext200/Dropbox/metis/p4_fletcher/pick/'

with open(path + 'cleaned_talks.pkl', 'rb') as picklefile:
    cleaned_talks = pickle.load(picklefile)

with open(path + 'ted_all.pkl', 'rb') as picklefile:
    ted_all = pickle.load(picklefile)

with open(path + 'lda_mod.pkl', 'rb') as picklefile:
    lda_mod = pickle.load(picklefile)

with open(path + 'lda_data.pkl', 'rb') as picklefile:
    lda_data = pickle.load(picklefile)

with open(path + 'vect_mod.pkl', 'rb') as picklefile:
    vect_mod = pickle.load(picklefile)

with open(path + 'topic_names.pkl', 'rb') as picklefile:
    topic_names= pickle.load(picklefile)

titles = ted_all['title']
#---------- MODEL IN MEMORY ----------------#
def get_recommendations(first_article, model, vectorizer, training_vectors,title, ind):

    new_vec = model.transform(
        vectorizer.transform([first_article]))

    nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    nn.fit(training_vectors)
    results = nn.kneighbors(new_vec)
    recommend_list = results[1][0]
    scores = results[0]

    rec_dict = defaultdict(list)

    ss = np.array(scores).flat
    for i, resp in enumerate(recommend_list):
        #print('--- ID ---\n', + resp)
        #print('--- dist ---\n', + ss[i])
        rec_dict["0"].append(ted_all.iloc[resp,1])

    rec_dict["0"].append(title)
    rec_dict["0"].append(topic_names.iloc[ind,0])
    return rec_dict


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, ted_rec.html
    """
    with open("ted_rec.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/ted", methods=["POST"])
def ted():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    print(data)
    X = data#["example"])
    #X = str(X[0])
    tite, score, talk_ind = process.extractOne(X, titles, scorer=fuzz.token_set_ratio)
    recs = get_recommendations(cleaned_talks[talk_ind],lda_mod, vect_mod, lda_data, tite,talk_ind)

    print(recs.values())
    # Put the result in a nice dict so we can send it as json
    #results = {"recommends": recs[1]}
    #print(results)
    return flask.jsonify(recs['0'])
    #return flask.json.dump(recs,fp)
#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)

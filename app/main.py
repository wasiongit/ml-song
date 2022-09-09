import flask
import io
import string
import time
import os
import pickle
import joblib
import numpy as np
import http.client
import json
# import tensorflow as tf
# from PIL import Image
from flask import Flask, jsonify, request
# from modelTraining import ms, encoder, random_forest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import platform

pathScaler = os.getcwd() + "\\app\scaler.save" if platform.system() == "Windows" else os.getcwd() + "/app/scaler.save"
pathEncoder = os.getcwd() + "\\app\classes.npy" if platform.system() == "Windows" else os.getcwd() + "/app/classes.npy"
pathModel = os.getcwd() + "\\app\\random_forest.sav" if platform.system() == "Windows" else os.getcwd() + "/app/random_forest.sav"
ms = joblib.load(pathScaler)

# model = tf.keras.models.load_model('resnet50_food_model')
encoder = LabelEncoder()
encoder.classes_ = np.load(pathEncoder, allow_pickle=True)

loaded_model = pickle.load(open(pathModel, 'rb'))

# def prepare_image(img):
#     img = Image.open(io.BytesIO(img))
#     img = img.resize((224, 224))
#     img = np.array(img)
#     img = np.expand_dims(img, 0)
#     return img


# def predict_result(img):
#     return 1 if model.predict(img)[0][0] > 0.5 else 0


app = Flask(__name__)


# @app.route('/predict', methods=['POST'])
# def infer_image():
#     if 'file' not in request.files:
#         return "Please try again. The Image doesn't exist"

def predict_mood(rDict):
    fList = ['duration_ms', 'danceability', 'acousticness', 'energy', 'instrumentalness',
             'liveness', 'valence', 'loudness', 'speechiness', 'tempo']
    OList = []
    for key in fList:
        OList.append(rDict[key])
    inp = OList
    inpT = ms.transform([inp])
    loaded_model.predict(inpT)
    prediction = encoder.classes_[loaded_model.predict(inpT)[0]]
    return prediction


@app.route("/predict", methods=["POST"])
def test():
    mood = request.json["mood"]
    access_token = request.json["access_token"]
    songsList = request.json["songlist"]
    reoccuring = request.json["reoccuring"]
    newSongList = []
    for id in songsList:
        conn = http.client.HTTPSConnection("api.spotify.com")
        payload = ''
        headers = {
            'Authorization': 'Bearer ' + access_token
        }
        conn.request("GET", "/v1/audio-features/" + id, payload, headers)
        res = conn.getresponse()
        data = res.read()
        analysis_data = json.loads(data.decode("utf-8"))
        try:
            errorCode = analysis_data['error']['status']
            if (errorCode == 404 or errorCode == 400):
                print("error")
                continue
        except:
            print("okay")
        prediction = predict_mood(analysis_data)
        if (prediction.upper() == mood.upper()):
            newSongList.append(id)

    playlist_id = None
    # finding if a playlist already exists
    try:
        conn = http.client.HTTPSConnection("api.spotify.com")
        payload = ''
        headers = {
            'Authorization': 'Bearer ' + access_token
        }
        conn.request("GET", "/v1/me/playlists", payload, headers)
        res = conn.getresponse()
        data = res.read()
        user_exists_playlists = json.loads(data.decode("utf-8"))['items']

        for item in user_exists_playlists:
            if (item['name'] == "ASORTA"):
                # found playlist with same name and unfollowing it
                if not reoccuring:
                    conn = http.client.HTTPSConnection("api.spotify.com")
                    payload = ''
                    headers = {
                        'Accept': 'application/json',
                        'Accept-Encoding': 'application/json',
                        'Authorization': 'Bearer ' + access_token
                    }
                    conn.request("DELETE", "/v1/playlists/" + item['id'] + "/followers", payload, headers)
                    res = conn.getresponse()
                    data = res.read()
                else:
                    playlist_id = item['id']
    except:
        print("no id")
    if (len(newSongList) > 0):
        # getting user id
        conn = http.client.HTTPSConnection("api.spotify.com")
        payload = ''
        headers = {
            'Authorization': 'Bearer ' + access_token
        }
        conn.request("GET", "/v1/me/", payload, headers)
        res = conn.getresponse()
        data = res.read()
        user_id = json.loads(data.decode("utf-8"))['id']

        if (not playlist_id):
            # create a playlist
            conn = http.client.HTTPSConnection("api.spotify.com")
            payload = json.dumps({
                "name": "ASORTA",
                "description": "New playlist description",
                "public": False
            })
            headers = {
                'Authorization': 'Bearer ' + access_token,
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/v1/users/" + user_id + "/playlists", payload, headers)
            res = conn.getresponse()
            data = res.read()
            mid_data = data.decode("utf-8")
            playlist_id = json.loads(data.decode("utf-8"))['id']

        # new playlist item
        # conn = http.client.HTTPSConnection("api.spotify.com")
        # payload = ''
        # headers = {
        # 'Accept': 'application/json',
        # 'Content-Type': 'application/json',
        # 'Authorization': 'Bearer '+access_token
        # }
        # conn.request("GET", "/v1/playlists/"+playlist_id+"/tracks", payload, headers)
        # res = conn.getresponse()
        # data = res.read()
        # try:
        #     response=json.loads(data.decode("utf-8"))
        #     def mapFn(a):
        #         return a['track']['id']
        #     song_id_list=list(map(mapFn,response['items']))
        #     oldSongList=song_id_list
        # except:
        #     print("new playlist")
    else:
        if playlist_id:
            return "success:" + playlist_id
        else:
            return "success:not_classified"
    # adding songs to playlist
    for index, song_id in enumerate(list(set(newSongList))):
        conn = http.client.HTTPSConnection("api.spotify.com")
        payload = ''
        headers = {
            'Authorization': 'Bearer ' + access_token,
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/playlists/" + playlist_id + "/tracks?position=" + str(
            index) + "&uris=spotify%3Atrack%3A" + song_id, payload, headers)
        res = conn.getresponse()
        data = res.read()
        response = json.loads(data.decode("utf-8"))
    return "success:" + playlist_id
    # rDict = request.json["input"]
    # return predict_mood(rDict)


@app.route("/model_result", methods=["POST"])
def test2():
    mood = request.json["mood"]
    access_token = request.json["access_token"]
    playlist_id = request.json["playlist"]
    songsList = []
    newSongList = []
    final_data = {}
    # new playlist item
    conn = http.client.HTTPSConnection("api.spotify.com")
    payload = ''
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + access_token
    }
    conn.request("GET", "/v1/playlists/" + playlist_id + "/tracks", payload, headers)
    res = conn.getresponse()
    data = res.read()
    try:
        response = json.loads(data.decode("utf-8"))

        def mapFn(a):
            return a['track']['id']

        song_id_list = list(map(mapFn, response['items']))
        songsList = song_id_list
    except:
        print("see")
    for id in songsList:
        conn = http.client.HTTPSConnection("api.spotify.com")
        payload = ''
        headers = {
            'Authorization': 'Bearer ' + access_token
        }
        conn.request("GET", "/v1/audio-features/" + id, payload, headers)
        res = conn.getresponse()
        data = res.read()
        analysis_data = json.loads(data.decode("utf-8"))
        prediction = predict_mood(analysis_data)
        conn = http.client.HTTPSConnection("api.spotify.com")
        payload = ''
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + access_token
        }
        conn.request("GET", "/v1/tracks?ids=" + id, payload, headers)
        res = conn.getresponse()
        data = res.read()
        name = json.loads(data.decode("utf-8"))['tracks'][0]['name']
        final_data[id] = {"result": prediction, "name": name}
    return final_data
    # rDict = request.json["input"]
    # return predict_mood(rDict)


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
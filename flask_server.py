from flask import Flask, request, render_template

import searchpy

import pandas as pd
foodOlist = pd.read_csv('foodOlist.csv', header=None)
foodXlist = pd.read_csv('foodXlist.csv', header=None)
imgadvX = pd.read_csv('imgadvX.csv', header=None)
imgadvX = imgadvX[0].values.tolist()
imgadvO = pd.read_csv('imgadvO.csv', header=None)
imgadvO = imgadvO[0].values.tolist()

from keras.models import load_model
model1 = load_model('food_cnn.h5')
model2 = load_model('like_cnn1000.h5')


app = Flask(__name__)
app.secret_key = 'acorn_you_can_not_this'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search_insta")
def search_insta():
    return render_template("search_insta.html")

@app.route("/search_url")
def search_url():
    return render_template("search_url.html")

@app.route("/search_upload")
def search_upload():
    return render_template("search_upload.html")

@app.route("/search_process_result", methods = ['GET'])
def search_process_result():
    picname = request.full_path.split('=')[1].split('_')[3]
    wordt1, wordl1, wordt2, wordl2, print1, print2, print3 = searchpy.process(request.full_path.split('=')[1],model1,model2)
    return render_template("search_process_result.html",picname = picname,wordt1=wordt1,wordl1=wordl1,wordt2=wordt2,wordl2=wordl2, print1=print1, print2=print2, print3=print3)

@app.route("/search_process", methods = ['GET'])
def search_process():
    from random import sample
    import os
    path_dir = 'static/sample'
    file_list = os.listdir(path_dir)
    imgdata = sample(file_list, 21)
    return render_template("search_process.html", imgdata = imgdata)

@app.route("/search_start", methods =["POST"])
def search_start():
    selection = request.form['selection']
    searchname = request.form['txt']
    return searchpy.estimate(searchname, selection, model1, model2)

@app.route("/url_search", methods =["POST"])
def url_search():
    imgurl = request.form['imgurl']
    return searchpy.urlsearch(imgurl, model1, model2)

@app.route("/file_upload", methods =["POST"])
def file_upload():
    file = request.files['file']
    if not file:
        return render_template("search_upload.html",no=1)
    file.save('static/upload/check.jpg')
    searchpy.uploadsearch(model1,model2)
    return render_template("search_upload_result.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False ,threaded=False)


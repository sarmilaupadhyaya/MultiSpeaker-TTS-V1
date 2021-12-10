# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:35 2021

@author: rasul

"""
from flask import Flask, render_template, request, send_file, send_from_directory
import os
from os.path import exists
from web_cpu_inf import main as inf

app = Flask(__name__)
#avoid using cached audio
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/", methods = ['POST', 'GET'])
def home():
    if request.method == "POST":
        language = request.form["text_lang"].lower()
        lang_codes = {"fr" : 0, "en": 1}
        lang_id = lang_codes[language]
        speaker_id = int(request.form["speaker"])
        #embedding reps don't work yet
        lang_rep = request.form["lang_rep"]
        speaker_rep = request.form["speaker_rep"]
        string = request.form["text"]
        diffusion = int(request.form["diffusion"])
        #accent
        inf(string, timesteps=diffusion, language=language,
            lang_id=lang_id, speaker_id=speaker_id, 
            speaker_rep=speaker_rep, lang_rep=lang_rep)
        return render_template("home.html")
    else:
        return render_template("home.html")
    
@app.route("/out/web/latest.wav", methods = ['GET'])
def read():
    return send_from_directory(directory="out/web/", filename="latest.wav", cache_timeout=0)


if __name__ == "__main__":
    app.run()
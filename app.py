# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:35 2021

@author: rasul

"""
from flask import Flask, render_template, request, send_from_directory
from web_cpu_inf import main as inf
import os, sys
from gtts import gTTS
app = Flask(__name__)
#avoid using cached audio
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

import kv_tts as kv
@app.route("/", methods = ['POST', 'GET'])
def home():
    if request.method == "POST":
        compare = request.form["gtts"]
        language = request.form["text_lang"].lower()
        lang_codes = {"fr" : 0, "kv" :0, "en": 1}
        lang_id = lang_codes[language]
        speaker_id = int(request.form["speaker"])
        #embedding reps don't work yet
        lang_rep = request.form["lang_rep"]
        speaker_rep = request.form["speaker_rep"]
        string = request.form["text"]
        diffusion = int(request.form["diffusion"])
        start = request.form["start"]
        #accent
        if language == "kv":
            string = kv.convert(string, start="kv", output="ipa")

        inf(string, timesteps=diffusion, language=language,
            lang_id=lang_id, speaker_id=speaker_id, 
            speaker_rep=speaker_rep, lang_rep=lang_rep)
        if compare:
            #placeholder
            tts = None
            if language == "kv":
                tts = kv.read_audio("kv", "fr", string)
                pass
            else:
                tts = gTTS(string, lang=language)
            kv.write_tts(tts, "out/web", "gt.mp3")
        return render_template("home.html")
    else:
        return render_template("home.html")
    
@app.route("/out/web/model1.wav", methods = ['GET'])
def model1():
    return send_from_directory(directory="out/web/", filename="model1.wav", cache_timeout=0)

@app.route("/out/web/gt.mp3", methods = ['GET'])
def gt():
    return send_from_directory(directory="out/web/", filename="gt1.wav", cache_timeout=0)

if __name__ == "__main__":
    app.run()

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:35 2021

@author: rasul

"""
from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS
from web_cpu_inf import main as inf
import os, sys
from gtts import gTTS
app = Flask(__name__)
CORS(app)
#avoid using cached audio
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
base=  os.path.dirname(os.path.abspath(__file__))

from kv_tts import convert, read_audio, write_tts
@app.route("/", methods = ['POST', 'GET'])
def home():
    checkpts = "MultiSpeaker-TTS-V1/models"
    
    if request.method == "POST":
        compare = request.form["compare"]
        language = request.form["text_lang"].lower()
        lang_codes = {"fr" : 0, "kv" :0, "en": 1}
        lang_id = lang_codes[language]
        speaker_id = int(request.form["speaker"])
        rep= request.form["rep"]
        #lang_rep = request.form["lang_rep"]
        #speaker_rep = request.form["speaker_rep"]
        string = request.form["text"]
        diffusion = int(request.form["diffusion"])
        out_f = "model1.wav"
        #accent
        if language == "kv":
            string = convert(string, start="kv", output="ipa")
            string = string.replace(" ", "")
            string = string.replace(",", " ")
            string = string.replace(".", "  ")

        inf(string, checkpts=checkpts, timesteps=diffusion, language=language,
            lang_id=lang_id, speaker_id=speaker_id, 
            rep=rep, out_f=out_f)
        if compare:
            #placeholder
            tts = None
            if language == "kv":
                tts = read_audio("kv", "fr", string)
            else:
                tts = gTTS(string, lang=language)
            write_tts(tts, "out/", "gt.mp3", base=base)
        return render_template("home.html")
    else:
        return render_template("home.html")
    
@app.route("out/model1.wav", methods = ['GET'])
def model1():
    return send_from_directory("out", "model1.wav")

@app.route("out/gt.mp3", methods = ['GET'])
def gt():
    return send_from_directory("out", "gt.mp3")

if __name__ == "__main__":
    app.debug = True
    app.run()

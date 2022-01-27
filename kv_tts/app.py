# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 10:30:35 2021

@author: rasul

"""
from flask import Flask, render_template, request, redirect, send_file, send_from_directory
import os.path as path
from gtts import gTTS
from transliterator import convert
import make_syllables as syl

app = Flask(__name__)
#avoid using cached audio
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
def read_audio(lang, speaker, text):
    basedir = path.abspath(path.dirname(__file__))
    folder = lang + "_" + speaker
    folder = path.join(basedir, folder)
    onsets, nuclei, codas = syl.load_folder(folder)
    converted = syl.get_pronunciation(text, onsets, nuclei, codas)
    tts = gTTS(converted, lang=speaker)
    return tts

def write_tts(tts, folder, f):
        basedir = path.abspath(path.dirname(__file__))
        #for some reason, there was a problem doing the join as one step
        out_dir = path.join(basedir, folder)
        out = path.join(out_dir, f)
        tts.save(out)
        
@app.route("/", methods = ['POST', 'GET'])
def home():
    if request.method == "POST":
        start = request.form["start"]
        into = request.form["into"]
        norm = request.form["norm"]
        string = request.form["text"]
        converted = convert(string, start=start, output=into, norm=norm)
        return render_template("home.html", output=converted)

    else:
        return render_template("home.html")

@app.route("/speak", methods = ['POST', 'GET'])
def speak():
    if request.method == "POST":
        speaker = request.form["speaker"].lower()
        start = request.form["start"]
        text = request.form["text"]
        ipa = convert(text, start=start, output="ipa")
        tts = read_audio("kv", speaker, ipa)
        write_tts(tts, "data", "read.mp3")
        return render_template("speak.html")

    else:
        return render_template("speak.html")

@app.route("/data/read.mp3", methods = ['GET'])
def read():
    return send_from_directory(directory="data", filename="read.mp3", cache_timeout=0)

if __name__ == "__main__":
    app.run()

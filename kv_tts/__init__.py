# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:47:44 2021

@author: rasul
"""

import re

def to_upper(r):
    return "".join([r.group(0)[:-1], r.group(3).upper()])

def normalize(text):
    text = text.replace("\n ", "\n")
    text = re.sub(r" (?=\w\-\s\n\w)", "\n", text)
    text = re.sub(r"(?<=\w)\-\s\n(?=\w)", "", text)
    text = text.replace("llon", "yon")
    text = text.replace("ngg", "ng")
    #tense markers
    text = re.sub("(?<=\W)s[eè]?r(?=[aé]\W)", "s", text)
    text = re.sub("(?<=\W)s[eèé](?=té\W)", "", text)
    #Contraction
    text = text.replace("oulé", "olé")
    text = re.sub(r"[aéio] apé", "'apé", text)
    text = re.sub(r"(?<=[MmTtLlYyNnPp])apé", "'apé", text)
    text = re.sub(r"(?<=\W[MmtT])a(?=\W)", "'a", text)
    text = re.sub(r"(?<=\W[tTmMnNlLYy])[(ou*)éi]\s*olé(?=\W|\Z)", "'olé", text)
    #agglutination and compounding
    text = re.sub(r"(?<=\W)d[oò] lo(?=\W*)", "dolo", text)
    text = re.sub(r"(?<=[Tt]ou)l*\s(?=(swi)|(tem)|(kèk)|(m[òo]u*n*))", "", text)
    text = re.sub(r"(?<=[Ll]a) (?=(tèt)|(mèzon)|(plansh)|(hash))", "", text)

    text = text.replace(" si la", "-çila")
    text = text.replace(" la yé", "-layé")
    #Undo optional nasalization
    text = re.sub(r"(?<=[\w][ñnm])in(?=\W|\Z)*", "é", text)
    #Drop some final consonants
    text = re.sub(r"(?<=s)t(?=\W|\Z)", "", text)
    text = re.sub(r"(?<=m)[p](?=\W|\Z)", "", text)
    text = re.sub(r"r(?=t\W|\Z)", "", text)
    text = re.sub(r"(?<=[bdgkpth])[lrɾ]e*(?=\W|\Z)", "", text)
    text = re.sub(r"(?<=n)d(?=\W|\Z)", "", text)
    text = re.sub(r"(?<=[b(ary)(mw)])en(?=\W|\Z)", "in", text)

    #First letter after punctuation
    text = re.sub(r"([.?!—])([\W])*(\w)", to_upper, text)
    return text
  
  # -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:07:11 2021

@author: rasul
"""
import epitran
import os

fr = epitran.Epitran("lou-Latn-fra")
kv_in =  epitran.Epitran("lou-IPA-kv")
kv_out = epitran.Epitran("lou-Latn-kv")
dlc_in =  epitran.Epitran("lou-IPA-dlc")
dlc_out = epitran.Epitran("lou-Latn-dlc")

def transliterate(string, out, into):
    forward = out.transliterate(string, normpunc=(True))
    back = into.transliterate(forward)
    return(back)


"""After transliterating, this function capitalizes the first letter of the transliteration
if the original word was also capitalized"""
def match_capitals(transliterated, reference):
    reference = reference.split(" ")
    transliterated = transliterated.split(" ")
    for i, word in enumerate(transliterated):
        if len(word) > 0 and i < len(reference) and reference[i][0].isupper():
            transliterated[i] = word[0].upper() + word[1:]
    return " ".join([str(word) for word in transliterated])


def fr_kv(string="", fr=fr, kv=kv_in, norm=True):
    out = transliterate(string, fr, kv)
    out = match_capitals(out, string)
    if norm:
        out = normalize(out)
    return out

def kv_ipa(string="", kv=kv_out, norm=False):
    return kv.transliterate(string)

def dlc_ipa(string="", dlc=dlc_out, norm=False):
    return dlc.transliterate(string)

def ipa_dlc(string="", ipa=dlc_in, norm=False):
    return ipa.transliterate(string)

def kv_dlc(string="", kv=kv_out, dlc=dlc_in, norm=False):
    if norm:
        string = normalize(string)
    inter = kv_ipa(string)
    return match_capitals(ipa_dlc(inter), string)

def fr_ipa(string="", fr=fr, ipa = kv_out, norm=True):
    inter = fr_kv(string, norm=norm)
    return kv_ipa(inter)

def fr_dlc(string="", fr=fr, ipa=kv_out, dlc=dlc_in, norm=True):
    first = fr_kv(string, norm=norm)
    second = kv_ipa(string=first)
    last = ipa_dlc(string=second)
    return match_capitals(last, first)

def ipa_kv(string="", kv=kv_in, norm=True):
    raw = kv.transliterate(string)
    if norm:
        return normalize(raw)
    else:
        return raw

def dlc_kv(string="", dlc=dlc_out, kv=kv_in, norm=True):
    inter = transliterate(string, dlc, kv)
    if norm:
        inter = normalize(inter)
    return match_capitals(inter, string)

def ident(string, norm=False):
    return string


def convert(string, start="fr", output="kv", norm="Y"):
    norm = norm == "Y"
    start = start.lower()
    output = output.lower()
    input_options = ["fr", "kv", "dlc", "ipa"]
    output_options = ["kv", "dlc", "ipa"]
    combinations = [[fr_kv, fr_dlc, fr_ipa],
                    [ident, kv_dlc, kv_ipa],
                    [dlc_kv, ident, dlc_ipa],
                    [ipa_kv, ipa_dlc, ident]]

    string =string.replace("\n ", "\n")
    #epitran gets overwhelmed by long strings so split and recombine
    splits = string.split("\n")
    out = ""
    for sub in splits:
        sub = sub + "\n"
        i = input_options.index(start)
        j = output_options.index(output)
        #restore the period we split on
        new = combinations[i][j](string=sub, norm=norm)
        if start == "kv" and output =="kv" and norm:
            new = normalize(new)
        out += new

    return out

import os.path as path
#b for backward, f for forward
def read_file(folder, fname):
    fname = path.join(folder, fname)
    dic = {}
    with open(fname, encoding="utf8") as f:
        data = f.read()
        lines = data.splitlines()
        for line in lines[1:]:
            line = line.split(",")
            dlc = line[0]
            fr = line[1]
            dic[dlc] = fr
            #print(dlc, fr)
    return dic


#match the longest string possible in part dic
def get_syllable_part(s, part_dic):
    part = ""
    if s:
        proposed = s[0]
        while proposed[::-1] in part_dic.keys() and len(s) > 1:
            part = proposed
            s = s[1:]
            proposed = proposed + s[0]
        if proposed[::-1] in part_dic.keys() and len(s) == 1:
            part = proposed
            s = ""
    return s, part[::-1]

def syllabify(s, onsets, nuclei, codas):
    s = handle_nasals(s)
    s = s[::-1]
    if s.isalpha():
        syllables = []
        while s:
            s, b_coda = get_syllable_part(s, codas)
            s, b_nucleus = get_syllable_part(s, nuclei)
            s, b_onset = get_syllable_part(s, onsets)
            syllable = [b_onset, b_nucleus, b_coda]
            syllables.append(syllable)
    else:
        syllables = [[""], [s], [""]]
    return syllables[::-1]


def transliterate_syl(trio, onsets, nuclei, codas):
    print(trio)
    ons, nuc, coda = trio[0], trio[1], trio[2]
    try:
        ons, nuc, coda = onsets[ons], nuclei[nuc], codas[coda]
    #don't do anything for undefined characters
    except:
        print("not in dic")
        pass
    if ons:
        nuc = nuc.replace("h", "")
    return "".join([ons, nuc, coda])

#reversing the nasals converts them into two characters and breaks the algorithm
def handle_nasals(word):
    word = word.replace("ɛ̃", "E")
    word = word.replace("œ̃", "E")
    word = word.replace("ɔ̃", "O")
    word = word.replace("ɑ̃", "A")
    return word

def clean_output(word, onsets, codas):
    letters = set(onsets.values()).union(set(codas.values()))
    letters = "".join(letters)
    sil_before_sound = "[hst]1(?=[" + letters + "]" +")"
    hard_c = "c(?=[ieé(\'on)]" +")"
    soft_j = "j(?=[a]" +")"

    #silent letters that shouldn't be pronounced
    #if there is a coda
    word = re.sub(sil_before_sound, "", word)
    word = re.sub(hard_c, "qu", word)
    word = re.sub(soft_j, "je", word)
    #clean up messy vowels
    word = word.replace("ii", "i")
    word = word.replace("yy", "i")
    word = word.replace("lye", "lié")
    word = word.replace("1", "")
    return word

def transliterate_word(word, onsets, nuclei, codas):
    transliterated = []
    syllables = syllabify(word,onsets, nuclei, codas)
    print(syllables)
    for syllable in syllables:
        transliterated.append(transliterate_syl(syllable, onsets, nuclei, codas))
    total = " ".join(transliterated)
    return clean_output(total, onsets, codas)

def load_folder(folder):
    onsets = read_file(folder,"onsets.txt")
    nuclei = read_file(folder,"nuclei.txt")
    codas = read_file(folder,"codas.txt")
    return onsets, nuclei, codas

def clean_phrase(phrase):
    #get rid of punctuation
    phrase = re.sub(r"[\"\-\\?\.\!,:—;]", "", phrase)
    #change normalize spacing
    phrase = re.sub(r"[\s*]", " ", phrase)
    #simplify consonant clusters
    phrase = re.sub(r"rw", "ɾ ", phrase)
    phrase = phrase.split(" ")
    #get rid of empty strings
    phrase = [w for w in phrase if len(w)]
    return phrase

def get_pronunciation(phrase, onsets, nuclei, codas):
    words = clean_phrase(phrase)
    pronunciations = []
    for word in words:
        pronunciation = transliterate_word(word, onsets, nuclei, codas)
        pronunciations.append(pronunciation)
    return " ".join(pronunciations)
  
from gtts import gTTS
def read_audio(lang, speaker, text):
    basedir = path.abspath(path.dirname(__file__))
    folder = lang + "_" + speaker
    folder = path.join(basedir, folder)
    onsets, nuclei, codas = load_folder(folder)
    converted = get_pronunciation(text, onsets, nuclei, codas)
    tts = gTTS(converted, lang=speaker)
    return tts

def write_tts(tts, folder, f, base=path.abspath(path.dirname(__file__))):
        #for some reason, there was a problem doing the join as one step
        out_dir = path.join(base, folder)
        out = path.join(out_dir, f)
        tts.save(out)

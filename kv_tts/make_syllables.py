# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:31:01 2021

@author: rasul
"""

import re
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

if __name__ == "__main__":
    folder = "kv_fr"
    onsets, nuclei, codas = load_folder(folder)
    test_s = transliterate_word("lapɛ̃", onsets, nuclei, codas)
    print(test_s)

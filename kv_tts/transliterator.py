# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:07:11 2021

@author: rasul
"""
import epitran
import re
import os
import kv_normalizer as normer

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


def fr_kv(string="", fr=fr, kv=kv_in, normalize=True):
    out = transliterate(string, fr, kv)
    out = match_capitals(out, string)
    if normalize:
        out = normer.normalize(out)
    return out

def kv_ipa(string="", kv=kv_out, normalize=False):
    return kv.transliterate(string)

def dlc_ipa(string="", dlc=dlc_out, normalize=False):
    return dlc.transliterate(string)

def ipa_dlc(string="", ipa=dlc_in, normalize=False):
    return ipa.transliterate(string)

def kv_dlc(string="", kv=kv_out, dlc=dlc_in, normalize=False):
    if normalize:
        string = normer.normalize(string)
    inter = kv_ipa(string)
    return match_capitals(ipa_dlc(inter), string)

def fr_ipa(string="", fr=fr, ipa = kv_out, normalize=True):
    inter = fr_kv(string, normalize=normalize)
    return kv_ipa(inter)

def fr_dlc(string="", fr=fr, ipa=kv_out, dlc=dlc_in, normalize=True):
    first = fr_kv(string, normalize=normalize)
    second = kv_ipa(string=first)
    last = ipa_dlc(string=second)
    return match_capitals(last, first)

def ipa_kv(string="", kv=kv_in, normalize=True):
    raw = kv.transliterate(string)
    if normalize:
        return normer.normalize(raw)
    else:
        return raw

def dlc_kv(string="", dlc=dlc_out, kv=kv_in, normalize=True):
    inter = transliterate(string, dlc, kv)
    if normalize:
        inter = normer.normalize(inter)
    return match_capitals(inter, string)

def ident(string, normalize=False):
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
        new = combinations[i][j](string=sub, normalize=norm)
        if start == "kv" and output =="kv" and norm:
            new =normer.normalize(new)
        out += new

    return out


if __name__ == "__main__":
    print(convert("Aujord'hui"))




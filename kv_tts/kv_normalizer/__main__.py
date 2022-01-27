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

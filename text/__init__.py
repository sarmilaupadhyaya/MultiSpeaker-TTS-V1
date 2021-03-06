""" from https://github.com/keithito/tacotron """

import re
import pickle
from text import cleaners
#from text.symbols import symbols
from feature_extraction import pipeline
import pandas as pd


#_symbol_to_id = {s: i for i, s in enumerate(symbols)}
#_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# saving symbol to id for inference as everytime we run this file, the order is changed.
with open('text/_symbol_to_id.pickle', 'rb') as handle:
    _symbol_to_id = pickle.load(handle)
    _id_to_symbol = {i: s for s,i in _symbol_to_id.items()}

#dff = pd.read_csv("text/final_phonemes.csv", sep="\t")[["symbol","id"]]
#_symbol_to_id = dict()
#for index, row in dff.iterrows():

#    _symbol_to_id[row["symbol"].replace("  "," ")] = row["id"]
#_id_to_symbol = {i:s for s,i in _symbol_to_id.items()}

def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word

df = pd.read_csv("text/kv_phonemes.csv", header=None)
df.columns=["ipa", "fr"]
kv_dict = {row["ipa"]:row["fr"] for index, row in df.iterrows()}


def text_to_sequence(text, cleaner_names=["english_cleaners"],arpabet_dict=None, dictionary=None, language="en"):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
    '''
    symbols = []
    sequence = []
    if language == "kv":
       print("converting KV")
       conversion = []
       #start with a string of IPA symbols (expand with converter later)
       for char in text:
           if char in kv_dict.keys():
               conversion.append(kv_dict[char])
           elif char in "??~":
               conversion[-1] = conversion[-1].lower() + "~"
       print(conversion)
       sequence = _symbols_to_sequence_kv(conversion)
       return sequence
    if language == "fr":
       print("converting French")
       p = pipeline.Preprocessing(text, dictionary, language="fr").get_sequence()
       print(p)

       for t in p:
           sequence += _symbols_to_sequence_french(t)
           symbols += t
       print(text)
       print(sequence)
       print(symbols)
       return sequence
    

    space = _symbols_to_sequence(' ')
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            if arpabet_dict is not None:
                clean_text = [get_arpabet(w, arpabet_dict) for w in clean_text.split(" ")]
                print(clean_text)
                for i in range(len(clean_text)):
                    t = clean_text[i]
                    symbols.append(t)
                    if t.startswith("{"):
                        
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)
                    sequence += space
            else:
                sequence += _symbols_to_sequence(clean_text)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
  
    # remove trailing space
    if dictionary is not None:
        print(sequence[-1])
        print(text)
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    print(symbols)
    print(text)
    print(sequence)
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    out =[]
    print(symbols)
    for i, s in enumerate(symbols):
        if  _should_keep_symbol(s) and not (i < len(symbols) -1 and "~" in symbols[i + 1]):
            out.append(_symbol_to_id[s])
        #these are one phoneme represented by two characters
        elif i < len(symbols) -1 and "~" in symbols[i + 1]:
            out.append(_symbol_to_id[s.lower() + "~"])    
    return out
    #return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    print(text)
    print("here")
    print(['@'+s for s in text.split()])
    return _symbols_to_sequence(['@' + s for s in text.split()])

def _symbols_to_sequence_french(phonemes):
    #return _symbols_to_sequence(['#' + s for s in phonemes]) 
    return [_symbol_to_id.get('#'+phonemes)]

def _symbols_to_sequence_kv(phonemes):
    return [_symbol_to_id[phoneme] for phoneme in phonemes]

def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'

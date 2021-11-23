""" from https://github.com/keithito/tacotron """

from text import cmudict
import pandas as pd


_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness:
_arpabet = ['@' + s for s in cmudict.valid_symbols]
_french_phonemes = pd.read_csv("text/phonemes.csv", header=None)
# Export all symbols:
symbols = list(set([_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + ["#" + x for x in _french_phonemes[0].tolist()]))

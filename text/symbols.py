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

symbs=[]
ids = []
start=38
for symbol in symbols:
    df = _french_phonemes[_french_phonemes[0] == symbol]
    if len(df) == 0:
        symbs.append(symbol)
        ids.append(start)
        start += 1
symbols=_french_phonemes[0].to_list() + symbs
new_ids = _french_phonemes[1].to_list() + ids
new_df = pd.DataFrame([], columns=["symbol", "id"])
new_df["symbol"] = pd.Series(symbols)
new_df["id"] = pd.Series(new_ids)
new_df.to_csv("new_phonemes.csv", sep="\t")


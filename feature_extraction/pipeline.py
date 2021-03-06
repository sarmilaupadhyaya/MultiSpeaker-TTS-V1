import subprocess
import os
import pandas as pd




class Preprocessing:

    """
    wrapper to convert french text to phonemes

    """

    def __init__(self,text, phoneme_dictionary, language="fr"):
        """

        """
        print("Input:", text)
        self.input_var = "feature_extraction/tmp/input.txt"
        self.output_var = "feature_extraction/tmp/output.txt"
        self.phoneme_out = "feature_extraction/tmp/phoneme_out.txt"
        self.text = text
        self.language=language
        with open(self.input_var, "w") as f:
            f.write(text)


    def integer_to_sequence(self, list_integer):
        pass

    def get_sequence(self):
        """

        """

        ## clean the french text if needed be
        ## put text in a temp file and get the output and selete the temp file
        ## run the pearl script
        if self.language=="fr":
            print("./get_phonemes.pl "+ self.input_var +" texts hts run > "+ self.output_var)
            pipe = subprocess.Popen(["./feature_extraction/get_phonemes.pl "+ self.input_var +" texts hts run > "+ self.output_var],stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True )
            output, errors = pipe.communicate()
            import pdb
            #pdb.set_trace()
            pipe = subprocess.Popen(["python3 feature_extraction/extract_phonemes.py  --input "+ self.output_var +" --output "+ self.phoneme_out],stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True )
            output, errors = pipe.communicate()

            #result = self.sequence_to_integer()
            with open(self.phoneme_out, 'r') as f:
                list_int = []
                phoneme = f.read().split(' ') 
            ## delete input and output file
            os.remove(self.input_var)
            os.remove(self.output_var)
            os.remove(self.phoneme_out)
            return phoneme

        else:
            each = list(self.text)
            df = pd.read_csv("../text/new_phonemes.csv", sep="\t")
            ids = []
            for e in each:
                try:
                    idd = df[df["symbol"]==e]["id"].tolist()[0]
                    ids.append(idd)
                except:
                    if e == '"':
                        ids.append(156)
                    elif e=='???':
                        ids.append(157)
                    elif e=='???':
                        ids.append(158)
                    else:

                        import pdb
                        #pdb.set_trace()
            assert len(each) == len(ids)
            return ids
                
                

        return result



        

        ## clean the output and get the sequence of phonemes
        ## convert phoneme sequence to id and return list of ids






#import pandas as pd

#df = pd.read_csv("phonemes.csv", header=None)
#df.columns=["phoneme", "id"]
#dictionary = {row["phoneme"]:row["id"] for index, row in df.iterrows()}
#from os import listdir
#from os.path import isfile, join
#mypath="/mnt/classes-so/Uni-of-Lorraine-Winter-Semester/Software-Project-2021-UL/Multispeaker-Grad-TTS_French_v1/SiwisFrenchSpeechSynthesisDatabase/text/part1"
#onlyfiles = [join(mypath, f) for f in listdir("/mnt/classes-so/Uni-of-Lorraine-Winter-Semester/Software-Project-2021-UL/Multispeaker-Grad-TTS_French_v1/SiwisFrenchSpeechSynthesisDatabase/text/part1") if isfile(join(mypath, f))]
#for each in onlyfiles:
#    text = open(each, "r").read()
#p = Preprocessing("Cela inclut notamment les communes de r??sidence de l???un ou l???autre des ??poux.", dictionary)
#result = p.get_sequence()
#print(result)

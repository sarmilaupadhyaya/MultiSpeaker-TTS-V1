FROM ubuntu
MAINTAINER Rasul Dent
RUN apt-get update && apt install tzdata -y
ENV TZ=Europe/Rome
RUN apt-get install -y git
RUN apt-get install -y python 3.7 && apt-get install -y python3-pip
RUN git clone https://github.com/sarmilaupadhyaya/MultiSpeaker-TTS-V1.git
RUN cd  MultiSpeaker-TTS-V1; pip3 install -r new_requirements.txt ;cd model/monotonic_align ;python3 setup.py build_ext --inplace
RUN git submodule init && git submodule update
RUN cd kv_tts/epitran; python3 setup.py install ; cd ../../..
RUN pip3 install gdown




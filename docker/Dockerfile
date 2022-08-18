FROM hpcaitech/colossalai:0.1.8

RUN mkdir /workspace && cd /workspace && git clone https://github.com/hpcaitech/EnergonAI.git --recursive && cd EnergonAI

RUN pip --no-cache-dir install -r requirements.txt && pip install .

WORKDIR /workspace

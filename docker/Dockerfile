FROM hpcaitech/colossalai:0.1.8
# FROM energonai:lastest

WORKDIR /workspace

RUN mkdir -p /workspace && cd /workspace && git clone https://github.com/hpcaitech/EnergonAI.git --recursive && cd EnergonAI && pip --no-cache-dir install -r requirements.txt && pip install .

CMD ["/workspace/EnergonAI/server.sh"]
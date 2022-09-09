FROM hpcaitech/colossalai:0.1.8

WORKDIR /workspace

RUN yum install -y vim
RUN mkdir -p /workspace && cd /workspace && git clone https://github.com/hpcaitech/EnergonAI.git --recursive && cd EnergonAI && pip --no-cache-dir install -r requirements.txt && pip install . && rm -rf /workspace/EnergonAI 

CMD ["bash", "/config/server.sh"]
FROM fkodom/flax:0.3.0-cuda11.1

RUN mkdir /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY flaxseed flaxseed/

ENTRYPOINT ["bash"]
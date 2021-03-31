FROM fkodom/flaxseed:0.1.0-cuda10.2

RUN apt-get update --fix-missing
RUN apt-get install -y git

RUN mkdir /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY train.py ./

CMD ["python", "train.py"]
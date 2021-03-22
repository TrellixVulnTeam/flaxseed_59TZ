FROM fkodom/flax:0.3.2-cuda10.2

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY flaxseed flaxseed/
COPY setup.py ./
COPY README.md ./
RUN python setup.py install

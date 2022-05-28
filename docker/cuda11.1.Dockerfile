FROM fkodom/deeplearning:py3.9-jax0.3-flax0.5-cuda11.1.1

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY flaxseed flaxseed/
COPY setup.py ./
COPY README.md ./
RUN python setup.py install

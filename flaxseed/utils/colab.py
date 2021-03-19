import requests
import os

from jax.config import config


def setup_tpu():
    if 'TPU_DRIVER_MODE' not in globals():
        addr = os.environ['COLAB_TPU_ADDR'].split(':')[0]
        url = f'http://{addr}:8475/requestversion/tpu_driver0.1-dev20191206'
        _ = requests.post(url)
        os.environ['TPU_DRIVER_MODE'] = 1

    config.FLAGS.jax_xla_backend = "tpu_driver"
    config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']

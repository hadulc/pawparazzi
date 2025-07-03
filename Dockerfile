FROM tensorflow/tensorflow:2.16.1

COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y libgl1
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY pawparazzi /pawparazzi

ARG model
ENV IN_CONTAINER=True
COPY models/${model} /models/model.keras

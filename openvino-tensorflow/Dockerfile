FROM ubuntu:18.04
USER root
RUN apt-get update \
  && apt-get install -y ffmpeg libsm6 libxext6 \
  && apt-get install -y python3.6 python3-pip
RUN pip3 install --upgrade pip==21.0.1 \
  && pip3 install --upgrade tensorflow-cpu==2.4.1 \
  && pip3 install openvino-tensorflow \
  && pip3 install opencv-python

RUN mkdir /app
RUN mkdir /app/result
WORKDIR /app
COPY openvino_tensorflow/examples /app/examples
COPY run.sh ./
RUN chmod +x run.sh
RUN chmod -R 777 /app
CMD [ "./run.sh" ]

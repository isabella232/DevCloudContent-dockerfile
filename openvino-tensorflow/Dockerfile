# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
FROM openvino/ubuntu20_runtime:2021.4.1
USER root
RUN apt-get update
RUN pip3 install --upgrade pip \
  && pip3 install tensorflow-cpu==2.5.1 \
  && pip3 install openvino-tensorflow==1.0.0 \
  && pip3 install opencv-python
RUN mkdir /app
RUN mkdir /app/result
WORKDIR /app
COPY openvino_tensorflow/examples /app/examples
COPY run.sh ./
RUN chmod +x run.sh
RUN chmod -R 777 /app
CMD [ "./run.sh" ]

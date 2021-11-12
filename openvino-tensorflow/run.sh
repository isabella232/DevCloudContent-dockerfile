#! /bin/bash
# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
echo "Running inference with OpenVINO-TensorFlow"
echo "Running inference with OpenVINO-TensorFlow" > /app/result/with_ov-tf.txt
python3 examples/classification_sample.py 2>&1 | tee -a /app/result/with_ov-tf.txt
echo "Running inference with TensorFlow"
echo "Running inference with TensorFlow" > /app/result/without_ov-tf.txt
python3 examples/classification_sample.py --disable_ovtf 2>&1 | tee -a /app/result/without_ov-tf.txt

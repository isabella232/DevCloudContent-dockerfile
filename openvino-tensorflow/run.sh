#! /bin/bash
echo "Running inference with OpenVINO-TensorFlow"
echo "Running inference with OpenVINO-TensorFlow" > /app/result/with_ov-tf.txt
python3 examples/classification_sample.py >> /app/result/with_ov-tf.txt
echo "Running inference with TensorFlow"
echo "Running inference with TensorFlow" > /app/result/without_ov-tf.txt
python3 examples/classification_sample.py --disable_ovtf >> /app/result/without_ov-tf.txt

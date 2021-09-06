import onnxruntime
import os
import os.path
import sys
import numpy
import time
import argparse
import cv2

parser = argparse.ArgumentParser(description='Using OpenVINO Execution Provider for ONNXRuntime')
parser.add_argument('--device', default='cpu', help="Device to perform inference on 'cpu (MLAS)' or on devices supported by OpenVINO-EP [CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16].")
parser.add_argument('--iters', help = "Number of iterations")
args = parser.parse_args()

onnxruntime.set_default_logger_severity(0) #Prints additional logger prints for easy debugging

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 1
model_path="resnet50-v1-12.onnx"
sess = onnxruntime.InferenceSession(model_path, sess_options)
print("\n")
print("Printing session providers: ")
print("\n")
print(sess.get_providers())
print("\n")

device = args.device

if(args.device == 'cpu'):
    print("Device type selected is 'cpu' which is the default CPU Execution Provider (MLAS)")
else:
    # Set OpenVINO as the Execution provider to infer this model
    sess.set_providers(['OpenVINOExecutionProvider'], [{'device_type' : device}])
    print("Device type selected is: {} using the OpenVINO Execution Provider".format(device))
    '''
    other 'device_type' options are: (Any hardware target can be assigned if you have the access to it)
    'CPU_FP32', 'GPU_FP32', 'GPU_FP16', 'MYRIAD_FP16', 'VAD-M_FP16'
    '''

print("Inputs info: ")
for elem in sess.get_inputs():
    print(elem)
print("\n")
    
print("Outputs info: ")
for elem in sess.get_outputs():
    print(elem)
print("\n")

input_shape = sess.get_inputs()[0].shape
print("Model input shape: ", input_shape)

img = cv2.imread("kitten.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(numpy.float32)
img = cv2.resize(img, (input_shape[3], input_shape[2]))
h_orig, w_orig, c = img.shape

mean_vec = numpy.array([0.485, 0.456, 0.406])
stddev_vec = numpy.array([0.229, 0.224, 0.225])
norm_img = (img/255 - mean_vec) / stddev_vec
norm_img = norm_img.transpose((2, 0, 1))
norm_img = norm_img.reshape(input_shape)
norm_img = numpy.asarray(norm_img, dtype='float32')


iters = int(args.iters)
ticks = time.time()
for i in range(iters): # 1 for running just 1 iteration
    res = sess.run([sess.get_outputs()[0].name], {'data': norm_img})[0]
ticks = time.time() - ticks
print("Inference time per frame: ", ticks/iters)
#print("res", res)
out = numpy.array(res)

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

#print("output: ", out)
#print("Output probabilities:", out.shape)
scores = numpy.squeeze(out)
a = numpy.argsort(scores)[::-1]
out_file = open("result.txt", "a")
for i in a[0:5]:
    print('class=%s ; probability=%f' %(labels[i],scores[i]))
    out_file.write('class=%s ; probability=%f\n' %(labels[i],scores[i]))
out_file.close()
import os
import urllib
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2, model_pb2

# https://stackoverflow.com/questions/42519010/how-to-do-batching-in-tensorflow-serving
# try method for batch prediction
# need changes on serving side too


from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--ip", required=True, dest="IP_ADDR", help="IP Address of TF Serve Endpoint")
parser.add_argument("--model-name", required=True, dest="MODEL_NAME", help="Deployed Model Name")
parser.add_argument("--image-path", required=True, dest="IMAGE_PATH", help="Input Image to be predicted")
parser.add_argument("--port", type=int, default=9000, dest="PORT", help="Deployed Service GRPC Port")

args = parser.parse_args()
IP_ADDR = args.IP_ADDR
MODEL_NAME = args.MODEL_NAME
IMAGE_PATH = args.IMAGE_PATH
PORT = int(args.PORT)

def make_request(stub, file_path, model_name):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    #request.model_spec.signature_name = 'serving_default'
    
    if file_path.startswith('http'):
        data = urllib.request.urlopen(file_path).read()
    else:
        with open(file_path, 'rb') as f:
            data = f.read()
    
    feature_dict = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()
    
    request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(serialized, shape=[1], dtype=tf.string))
    
    result_future = stub.Predict.future(request, 10.0)
    prediction = result_future.result()

    # print(prediction.outputs['classes'].int64_val)
    # print(prediction.outputs['probabilities'].float_val)

    pred_class = (prediction.outputs['classes'].int64_val)[0]
    pred_probs = prediction.outputs['probabilities'].float_val
    pred_class_prob = pred_probs[pred_class]
    
    return pred_class, pred_class_prob
    

channel = implementations.insecure_channel(IP_ADDR, PORT)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

dog_path = os.path.expanduser(IMAGE_PATH)
output = make_request(stub, dog_path, MODEL_NAME)
print(output)

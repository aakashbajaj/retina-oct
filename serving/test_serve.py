import os
import urllib
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2, model_pb2

def make_request(stub, file_path):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
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

    print(prediction.outputs)
    
    predicted_classes = list(zip(prediction.outputs['classes'].int64_val, prediction.outputs['probabilities'].float_val))

    print(predicted_classes)
    
    predicted_classes = list(reversed(sorted(predicted_classes, key = lambda p: p[1])))
    
    return predicted_classes

channel = implementations.insecure_channel('35.243.223.246', 8500)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

dog_path = os.path.expanduser('/home/aakashbajaj5311/datasets/OCT2017/test/CNV/CNV-538779-1.jpeg')
output = make_request(stub, dog_path)
print(output)

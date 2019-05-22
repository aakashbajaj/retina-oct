import os
import urllib
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2, model_pb2

def make_request(stub, file_path):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'retia'
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

    print(prediction.outputs['classes'].int64_val)
    print(prediction.outputs['probabilities'].float_val)

    pred_class = (prediction.outputs['classes'].int64_val)[0]
    pred_probs = prediction.outputs['probabilities'].float_val
    pred_class_prob = pred_probs[pred_class]
    
    return pred_class, pred_class_prob
    

channel = implementations.insecure_channel('34.73.240.123', 9000)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

dog_path = os.path.expanduser('/home/techno/oct_data/NORMAL-2362579-1.jpeg')
output = make_request(stub, dog_path)
print(output)

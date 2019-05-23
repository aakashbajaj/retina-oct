import kfp.dsl as dsl
import kfp.gcp as gcp


@dsl.pipeline(
  name='Retinal_OCT',
  description='Retinal OCT detection'
)
def dp_inf_pipe(
  # Important Parameters on top

  out_dir: dsl.PipelineParam = dsl.PipelineParam(name='data-dir', value='GCS_TFRECORD_OUTDIR_HERE'),
  model_dir: dsl.PipelineParam = dsl.PipelineParam(name='model-dir', value='MODEL_CHECKPOINT_DIR_HERE'),
  save_model_dir: dsl.PipelineParam = dsl.PipelineParam(name='save-model-dir', value="DIR_TO_EXPORT_SAVED_MODEL"),
  model_name: dsl.PipelineParam = dsl.PipelineParam(name='model-name', value='MODEL_NAME_FOR_SERVING'),


  # runner: dsl.PipelineParam = dsl.PipelineParam(name='runner', value=""),
  height: dsl.PipelineParam = dsl.PipelineParam(name='height', value=256),
  width: dsl.PipelineParam = dsl.PipelineParam(name='width', value=256),
  channels: dsl.PipelineParam = dsl.PipelineParam(name='channels', value=1),
  
  eval_steps: dsl.PipelineParam = dsl.PipelineParam(name='eval-steps', value=10000),
  epochs: dsl.PipelineParam = dsl.PipelineParam(name='num-epochs', value=1),
  batch_size: dsl.PipelineParam = dsl.PipelineParam(name='batch-size', value=32 ),
  max_train_steps: dsl.PipelineParam = dsl.PipelineParam(name='max-train-steps', value=10000),
  prefetch_buffer_size: dsl.PipelineParam = dsl.PipelineParam(name='prefetch-buffer', value=-1),

  # same as save_model_dir
  # model_path: dsl.PipelineParam = dsl.PipelineParam(name='model-path', value='MODEL_PATH'),
  num_gpus_serve: dsl.PipelineParam = dsl.PipelineParam(name='num-gpus-serve', value=0),

#   pred_inp_dir: dsl.PipelineParam = dsl.PipelineParam(name='pred_inp_dir', value='INPUT DIRECTORY FOR PREDICTION'),
#   model_location: dsl.PipelineParam = dsl.PipelineParam(name='model_location', value='TRAINED_MODEL_LOCATION'),
#   inf_batch_size: dsl.PipelineParam = dsl.PipelineParam(name='inf_batch_size', value=10)
):

  train = dsl.ContainerOp(
    name='train',
    image='gcr.io/speedy-aurora-193605/cnn_train_dis:latest',
    arguments=["--conv-dir", out_dir,
        "--model-dir", model_dir,
        "--save-model-dir", save_model_dir,
        "--num-epochs", epochs,
        "--batch-size", batch_size,
        "--max-train-steps", max_train_steps,
        "--eval-steps", eval_steps,
        # "--label-list", label_list_location,
        "--prefetch-buffer", prefetch_buffer_size,
        "--height", height,
        "--width", width,
        "--channels", channels,
        ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

  tensorbaord = dsl.ContainerOp(
    name='tensorbaord',
    image='gcr.io/speedy-aurora-193605/model-tensorbaord:latest',
    arguments=["--model-dir", model_dir,
      ],
      # file_outputs={'output': '/tmp/output'}

      ).apply(gcp.use_gcp_secret('user-gcp-sa'))  

  tfserve = dsl.ContainerOp(
    name='tfserve',
    image='gcr.io/speedy-aurora-193605/retina-tfserve:latest',
    arguments=["--model_name", model_name,
      "--model_path", save_model_dir,
      "--num_gpus", num_gpus_serve,
      ],
      # file_outputs={'output': '/tmp/output'}

      ).apply(gcp.use_gcp_secret('admin-gcp-sa'))
      
  train.set_gpu_limit('2')
  train.set_memory_request('8G')
  train.set_cpu_request('4')
  tfserve.after(train)

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(dp_inf_pipe,  'train_test_4cpu_2gpu_8gb.tar.gz')
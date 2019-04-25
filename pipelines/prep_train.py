import kfp.dsl as dsl
import kfp.gcp as gcp


@dsl.pipeline(
  name='Data Preparation',
  description='Preprocessing Image Data'
)
def dp_inf_pipe(
  inp_dir: dsl.PipelineParam = dsl.PipelineParam(name='input-dir', value='YOUR_GCS_INPDIR_HERE'),
  out_dir: dsl.PipelineParam = dsl.PipelineParam(name='data-dir', value='YOUR_GCS_OUTDIR_HERE'),
  num_shards: dsl.PipelineParam = dsl.PipelineParam(name='num-shards', value=2),
  split_flag: dsl.PipelineParam = dsl.PipelineParam(name='split-flag', value=2),
  train_split: dsl.PipelineParam = dsl.PipelineParam(name='train-split', value=0.8),
  seed: dsl.PipelineParam = dsl.PipelineParam(name='seed', value=123),
  height: dsl.PipelineParam = dsl.PipelineParam(name='height', value=224),
  width: dsl.PipelineParam = dsl.PipelineParam(name='width', value=224),
  fraction: dsl.PipelineParam = dsl.PipelineParam(name='fraction', value=1),

  model_dir: dsl.PipelineParam = dsl.PipelineParam(name='model-dir', value='YOUR_GCS_MODEL_DIR_HERE'),
  epochs: dsl.PipelineParam = dsl.PipelineParam(name='num-epochs', value=1),
  batch_size: dsl.PipelineParam = dsl.PipelineParam(name='batch-size', value=64),
  train_steps: dsl.PipelineParam = dsl.PipelineParam(name='train-steps', value=10000),
  prefetch_buffer_size: dsl.PipelineParam = dsl.PipelineParam(name='prefetch-buffer', value=None),
  label_list_location: dsl.PipelineParam = dsl.PipelineParam(name='label_list_location', value='JSON_FILE_CONTAINING_LABELS'),

#   pred_inp_dir: dsl.PipelineParam = dsl.PipelineParam(name='pred_inp_dir', value='INPUT DIRECTORY FOR PREDICTION'),
#   model_location: dsl.PipelineParam = dsl.PipelineParam(name='model_location', value='TRAINED_MODEL_LOCATION'),
#   inf_batch_size: dsl.PipelineParam = dsl.PipelineParam(name='inf_batch_size', value=10)
  ):

  data_prep = dsl.ContainerOp(
      name='data_prep',
      image='gcr.io/speedy-aurora-193605/oct_prep:v1',
      arguments=["--inp-dir", inp_dir,
          "--out-dir", out_dir,
          "--num-shards", num_shards,
          "--split-flag", split_flag,
          "--train-split", train_split,
          "--seed", seed,
          "--height", height,
          "--width", width,
          "--fraction", fraction
          ],
      # file_outputs={'output': '/tmp/output'}

      ).apply(gcp.use_gcp_secret('user-gcp-sa'))

  # data_prep.set_gpu_limit(2)
  # data_prep.set_memory_request('G')
  # data_prep.set_cpu_request('2')

  train = dsl.ContainerOp(
      name='train',
        image='gcr.io/speedy-aurora-193605/oct_train:v1',
        arguments=["--tfr-dir", out_dir,
            "--model-dir", model_dir,
            "--label-list", label_list_location,
            "--epochs", epochs,
            "--batch", batch_size,
            "--train-steps", train_steps,
            "--prefetch", prefetch_buffer_size,
            "--height", height,
            "--width", width,
          
        ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

  train.set_gpu_request('2')

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(dp_inf_pipe, __file__[:-3] + '_gpu.tar.gz')
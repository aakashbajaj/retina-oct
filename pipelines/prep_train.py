import kfp.dsl as dsl
import kfp.gcp as gcp


@dsl.pipeline(
  name='Retinal_OCT',
  description='Retinal OCT detection'
)
def dp_inf_pipe(
  inp_dir: dsl.PipelineParam = dsl.PipelineParam(name='input-dir', value='YOUR_GCS_INPDIR_HERE'),
  out_dir: dsl.PipelineParam = dsl.PipelineParam(name='data-dir', value='YOUR_GCS_OUTDIR_HERE'),
  project_id: dsl.PipelineParam = dsl.PipelineParam(name='project-id', value="YOUR_PROJECT_ID"),
  # runner: dsl.PipelineParam = dsl.PipelineParam(name='runner', value=""),
  num_shards: dsl.PipelineParam = dsl.PipelineParam(name='num-shards', value=5),
  split_flag: dsl.PipelineParam = dsl.PipelineParam(name='split-flag', value=2),
  train_split: dsl.PipelineParam = dsl.PipelineParam(name='train-split', value=0.8),
  seed: dsl.PipelineParam = dsl.PipelineParam(name='seed', value=123),
  height: dsl.PipelineParam = dsl.PipelineParam(name='height', value=256),
  width: dsl.PipelineParam = dsl.PipelineParam(name='width', value=256),
  channels: dsl.PipelineParam = dsl.PipelineParam(name='channels', value=1),

  model_dir: dsl.PipelineParam = dsl.PipelineParam(name='model-dir', value='YOUR_GCS_MODEL_DIR_HERE'),
  save_model_dir: dsl.PipelineParam = dsl.PipelineParam(name='save-model-dir', value=""),
  eval_steps: dsl.PipelineParam = dsl.PipelineParam(name='eval-steps', value=10000),
  epochs: dsl.PipelineParam = dsl.PipelineParam(name='num-epochs', value=1),
  batch_size: dsl.PipelineParam = dsl.PipelineParam(name='batch-size', value=64),
  max_train_steps: dsl.PipelineParam = dsl.PipelineParam(name='max-train-steps', value=10000),
  prefetch_buffer_size: dsl.PipelineParam = dsl.PipelineParam(name='prefetch-buffer', value=-1),

#   pred_inp_dir: dsl.PipelineParam = dsl.PipelineParam(name='pred_inp_dir', value='INPUT DIRECTORY FOR PREDICTION'),
#   model_location: dsl.PipelineParam = dsl.PipelineParam(name='model_location', value='TRAINED_MODEL_LOCATION'),
#   inf_batch_size: dsl.PipelineParam = dsl.PipelineParam(name='inf_batch_size', value=10)
):

  dataprep = dsl.ContainerOp(
    name='dataprep',
    image='gcr.io/speedy-aurora-193605/prep_tfr_df:v1',
    arguments=["--input-dir", inp_dir,
      "--output-dir", out_dir,
      "--num-shards", num_shards,
      "--split-flag", split_flag,
      "--train-split", train_split,
      "--project-id", project_id,
      # "--runner", runner,
      "--seed", seed,
      "--height", height,
      "--width", width,
      ],
      # file_outputs={'output': '/tmp/output'}

      ).apply(gcp.use_gcp_secret('user-gcp-sa'))

  # oct_data_prep.set_gpu_limit(2)
  # oct_data_prep.set_memory_request('G')
  # dataprep.set_cpu_request('2')

  train = dsl.ContainerOp(
    name='train',
    image='gcr.io/speedy-aurora-193605/cnn_train_dis:v2',
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
      
  train.set_gpu_limit('2')
  train.after(dataprep)

if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(dp_inf_pipe, __file__[:-3] + 'cnn_oct.tar.gz')
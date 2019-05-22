import kfp.dsl as dsl
import kfp.gcp as gcp


@dsl.pipeline(
  name='Retinal_OCT TF serve test',
  description='Retinal OCT detection'
)
def dp_inf_pipe(
  model_name: dsl.PipelineParam = dsl.PipelineParam(name='model-name', value='MODEL_NAME'),
  model_path: dsl.PipelineParam = dsl.PipelineParam(name='model-path', value='MODEL_PATH'),
  num_gpus: dsl.PipelineParam = dsl.PipelineParam(name='num-gpus', value=0),


#   pred_inp_dir: dsl.PipelineParam = dsl.PipelineParam(name='pred_inp_dir', value='INPUT DIRECTORY FOR PREDICTION'),
#   model_location: dsl.PipelineParam = dsl.PipelineParam(name='model_location', value='TRAINED_MODEL_LOCATION'),
#   inf_batch_size: dsl.PipelineParam = dsl.PipelineParam(name='inf_batch_size', value=10)
):

  dataprep = dsl.ContainerOp(
    name='tfserve',
    image='gcr.io/speedy-aurora-193605/retina-tfserve:latest',
    arguments=["--model_name", model_name,
      "--model_path", model_path,
      "--num_gpus", num_gpus,
      ],
      # file_outputs={'output': '/tmp/output'}

      ).apply(gcp.use_gcp_secret('admin-gcp-sa'))

  # oct_data_prep.set_gpu_limit()
  # oct_data_prep.set_memory_request('G')
  # dataprep.set_cpu_request('2')


if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(dp_inf_pipe, 'tfserve_test_pipe.tar.gz')
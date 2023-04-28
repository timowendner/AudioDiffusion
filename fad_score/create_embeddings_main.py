# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Uses Python Beam to compute the multivariate Gaussian."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from fad_score import create_embeddings_beam

ModelConfig = collections.namedtuple(
    'ModelConfig', 'model_ckpt embedding_dim step_size')

def main(input_file_list_path:str, output_path:str):
  pipeline = create_embeddings_beam.create_pipeline(
      tfrecord_input = None,
      files_input_list=input_file_list_path,
      feature_key='audio/reference/raw_audio',
      embedding_model=ModelConfig(
          model_ckpt='data/vggish_model.ckpt',
          embedding_dim=128,
          step_size=8000),
      embeddings_output=None,
      stats_output=output_path,
      name=output_path)
  result = pipeline.run()
  result.wait_until_finish()


if __name__ == '__main__':
  main()

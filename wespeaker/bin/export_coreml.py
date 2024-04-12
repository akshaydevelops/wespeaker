# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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

from __future__ import print_function

import argparse
import os

import torch
import yaml
import coremltools as ct

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


@register_torch_op()
def var(context, node):
    inputs = _get_inputs(context, node, expected=4)
    x = inputs[0]
    axes = inputs[1].val

    assert isinstance(axes, list) and all(
        isinstance(axis, int) for axis in axes)

    # Assert we can have biased divisor (N). (Change #1)
    assert inputs[2].val is False

    keepdim = True  # Set keepdim to True for broadcasting (Change #2)

    x_mean = mb.reduce_mean(x=x, axes=axes, keep_dims=keepdim)
    x_sub_mean = mb.sub(x=x,
                        y=x_mean)  # Broadcasting should work here (Change #4)
    x_sub_mean_square = mb.square(x=x_sub_mean)
    x_var = mb.reduce_mean(x=x_sub_mean_square, axes=axes, keep_dims=keepdim)

    context.add(x_var, torch_name=node.name)


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', required=True, help='output file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    print("model Loaded successfully")

    load_checkpoint(model, args.checkpoint)
    model.eval()

    example_input = torch.rand(1, 200, 80)
    traced_model = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)],
        minimum_deployment_target=ct.target.iOS13)
    mlmodel.save(args.output_file)

    print('Export model successfully, see {}'.format(args.output_file))


if __name__ == '__main__':
    main()

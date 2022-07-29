#!/usr/bin/env python
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import sys
import queue
import uuid
import asyncio

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


async def async_stream_yield(values, batch_size, sequence_id, model_name,
                             model_version):
    count = 1
    for value in values:
        # Create the tensor for INPUT
        value_data = np.full(shape=[batch_size, 1],
                             fill_value=value,
                             dtype=np.int32)
        inputs = []
        inputs.append(grpcclient.InferInput('INPUT', value_data.shape, "INT32"))
        # Initialize the data
        inputs[0].set_data_from_numpy(value_data)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('OUTPUT'))
        # Issue the asynchronous sequence inference.
        yield {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": '{}_{}'.format(sequence_id, count),
            "sequence_id": sequence_id,
            "sequence_start": (count == 1),
            "sequence_end": (count == len(values))
        }
        count = count + 1


async def main(FLAGS):

    # We use custom "sequence" models which take 1 input
    # value. The output is the accumulated value of the inputs. See
    # src/custom/sequence.
    int_sequence_model_name = "simple_dyna_sequence" if FLAGS.dyna else "simple_sequence"
    string_sequence_model_name = "simple_string_dyna_sequence" if FLAGS.dyna else "simple_sequence"
    model_version = ""
    batch_size = 1

    values = [11, 7, 5, 3, 2, 0, 1]

    # Will use two sequences and send them asynchronously. Note the
    # sequence IDs should be non-zero because zero is reserved for
    # non-sequence requests.
    int_sequence_id0 = 1000 + FLAGS.offset * 2
    int_sequence_id1 = 1001 + FLAGS.offset * 2

    # For string sequence IDs, the dyna backend requires that the
    # sequence id be decodable into an integer, otherwise we'll use
    # a UUID4 sequence id and a model that doesn't require corrid
    # control.
    string_sequence_id0 = str(1002 +
                              FLAGS.offset) if FLAGS.dyna else str(uuid.uuid4())

    int_result0_list = []
    int_result1_list = []

    string_result0_list = []

    async with grpcclient.InferenceServerClient(
            url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:

        # Request iterator that yields the next request
        async def async_request_iterator():
            async for request in async_stream_yield([0] + values, batch_size,
                                                    int_sequence_id0,
                                                    int_sequence_model_name,
                                                    model_version):
                yield request
            async for request in async_stream_yield(
                [100] + [-1 * val for val in values], batch_size,
                    int_sequence_id1, int_sequence_model_name, model_version):
                yield request
            async for request in async_stream_yield(
                [20] + [-1 * val for val in values], batch_size,
                    string_sequence_id0, string_sequence_model_name,
                    model_version):
                yield request

        try:
            # Start streaming
            response_iterator = triton_client.stream_infer(
                inputs_iterator=async_request_iterator(),
                stream_timeout=FLAGS.stream_timeout)
            # Read response from the stream
            user_data = UserData()
            async for response in response_iterator:
                result, error = response
                if error:
                    user_data._completed_requests.put(error)
                else:
                    user_data._completed_requests.put(result)
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

        # Retrieve results
        recv_count = 0
        while recv_count < (3 * (len(values) + 1)):
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                print(data_item)
                sys.exit(1)
            else:
                try:
                    this_id = data_item.get_response().id.split('_')[0]
                    if int(this_id) == int_sequence_id0:
                        int_result0_list.append(data_item.as_numpy('OUTPUT'))
                    elif int(this_id) == int_sequence_id1:
                        int_result1_list.append(data_item.as_numpy('OUTPUT'))
                    elif this_id == string_sequence_id0:
                        string_result0_list.append(data_item.as_numpy('OUTPUT'))
                    else:
                        print(
                            "unexpected sequence id returned by the server: {}".
                            format(this_id))
                        sys.exit(1)
                except ValueError:
                    string_result0_list.append(data_item.as_numpy('OUTPUT'))

            recv_count = recv_count + 1

    # Check results
    for i in range(len(int_result0_list)):
        int_seq0_expected = 1 if (i == 0) else values[i - 1]
        int_seq1_expected = 101 if (i == 0) else values[i - 1] * -1

        # For string sequence ID we are testing two different backends
        if i == 0 and FLAGS.dyna:
            string_seq0_expected = 20
        elif i == 0 and not FLAGS.dyna:
            string_seq0_expected = 21
        elif i != 0 and FLAGS.dyna:
            string_seq0_expected = values[i - 1] * -1 + int(
                string_result0_list[i - 1][0][0])
        else:
            string_seq0_expected = values[i - 1] * -1

        # The dyna_sequence custom backend adds the correlation ID
        # to the last request in a sequence.
        if FLAGS.dyna and (i != 0) and (values[i - 1] == 1):
            int_seq0_expected += int_sequence_id0
            int_seq1_expected += int_sequence_id1
            string_seq0_expected += int(string_sequence_id0)

        print("[" + str(i) + "] " + str(int_result0_list[i][0][0]) + " : " +
              str(int_result1_list[i][0][0]) + " : " +
              str(string_result0_list[i][0][0]))

        if ((int_seq0_expected != int_result0_list[i][0][0]) or
            (int_seq1_expected != int_result1_list[i][0][0]) or
            (string_seq0_expected != string_result0_list[i][0][0])):
            print("[ expected ] " + str(int_seq0_expected) + " : " +
                  str(int_seq1_expected) + " : " + str(string_seq0_expected))
            sys.exit(1)

    print("PASS: grpc aio sequence stream")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8001',
        help='Inference server URL and it gRPC port. Default is localhost:8001.'
    )
    parser.add_argument('-t',
                        '--stream-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Stream timeout in seconds. Default is None.')
    parser.add_argument('-d',
                        '--dyna',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Assume dynamic sequence model')
    parser.add_argument('-o',
                        '--offset',
                        type=int,
                        required=False,
                        default=0,
                        help='Add offset to sequence ID used')
    FLAGS = parser.parse_args()
    asyncio.run(main(FLAGS))

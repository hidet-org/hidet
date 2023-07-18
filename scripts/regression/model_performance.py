import os
import json
import numpy as np
import torch
import hidet
import argparse
from result_entry import ResultEntry, ResultGroup, load_regression_data

device_name = str(hidet.cuda.properties().name, 'UTF-8')

def bert_regression():
    regression_data = load_regression_data()
    result_group = ResultGroup(name='bert-base Regression', device_name=device_name)
    bert_data = regression_data[device_name]['bert_base_shapes']
    for shape, perf in bert_data.items():
        for dtype, ref_latency in perf.items():
            (m, n, k) = [int(s.strip()) for s in shape.split(',')]
            latency = 99999
            if not np.allclose(latency, ref_latency, rtol=0.05):
                # ToDo: deal with slowdown/speedup
                pass
            result_group.add_entry(ResultEntry(shape, dtype, latency))
    return result_group

def resnet_regression():
    # ToDo
    return None

def efficientnet_regression():
    # ToDo
    return None

def llama_regression():
    # ToDo
    return None

def model_performance_regression(report_file):
    result_groups = []
    result_groups.append(bert_regression())
    result_groups.append(resnet_regression())
    result_groups.append(efficientnet_regression())
    result_groups.append(llama_regression())
    with open(report_file, 'w') as f:
        f.write("---------------- Model Performance Regression -----------------\n")
        for result_group in result_groups:
            if result_group is not None:
                f.write(str(result_group))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Model Performance Regression')
    parser.add_argument(
        '--report',
        type=str,
        default='./report_model_performance.txt',
        help='Specify report output path'
    )
    args = parser.parse_args()
    model_performance_regression(args.report)
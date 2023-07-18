import json

lat = 99999
devices = ['NVIDIA GeForce RTX 3090', 'NVIDIA A100', 'NVIDIA A40']

# [M, N, K]
matmul_shapes = {
    '512, 512, 512': {},
    '1024, 1024, 1024': {},
    '2048, 2048, 2048': {},
    '4096, 4096, 4096': {},
    '8192, 8192, 8192': {},
    '16, 1024, 1024': {},
    '16, 4096, 4096': {},
    '16, 8192, 8192': {},
    '64, 1024, 1024': {},
    '64, 4096, 4096': {},
    '64, 8192, 8192': {},
    '1024, 64, 1024': {},
    '4096, 64, 4096': {},
    '8192, 64, 8192': {},
    '8192, 8192, 8176': {},
}
# [seqlen_q, seqlen_kv, hdim]
fmha_shapes = {
    '4096, 4096, 64': {},
    '4096, 4096, 128': {},
    '2048, 2048, 64': {},
    '2048, 2048, 128': {},
    '1024, 1024, 64': {},
    '1024, 1024, 128': {},
}

data = {'matmul_shapes': matmul_shapes, 'fmha_shapes': fmha_shapes}

data = {device:data for device in devices}

for gpu in devices:
    for shape in data[gpu]['matmul_shapes']:
        data[gpu]['matmul_shapes'][shape]['float16'] = lat
        data[gpu]['matmul_shapes'][shape]['float32'] = lat
    for shape in data[gpu]['fmha_shapes']:
        data[gpu]['fmha_shapes'][shape]['float16'] = lat

with open('regression_data.json', 'w') as f:
    json.dump(data, f, indent=2)
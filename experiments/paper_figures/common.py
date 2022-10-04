exec_color = {
    'torch': '#E08FFF',
    'ort': '#FF8F8F',
    'autotvm': '#6DBEFF',
    'ansor': '#54C45E',
    'trt': '#76B900',
    'hidet': '#FC9432',
}

# exec_edge_color = {
#     'torch': '#BA23F6',
#     'ort': '#F81313',
#     'autotvm': '#1071E5',
#     'ansor': '#008A0E',
#     'trt': '#518000',
#     'hidet': '#CC4E00',
# }
exec_common_edge_color = '#282C33'
exec_edge_color = {
    'torch': exec_common_edge_color,
    'ort': exec_common_edge_color,
    'autotvm': exec_common_edge_color,
    'ansor': exec_common_edge_color,
    'trt': exec_common_edge_color,
    'hidet': exec_common_edge_color,
}

hline_color = '#4C535D'
hline_alpha = 0.5

exec_fullname = {
    'torch': 'PyTorch',
    'ort': 'OnnxRuntime',
    'autotvm': 'AutoTVM',
    'ansor': 'Ansor',
    'trt': 'TensorRT',
    'hidet': 'Hidet',
}

end2end_data = {
    'torch': [4.767, 8.509, 4.023, 5.649, 5.742],
    'ort': [2.177, 5.132, 1.118, 2.891, 3.354],
    'autotvm': [1.785, 3.119, 0.376, 26.916, 40.895],
    'ansor': [1.488, 2.598, 0.306, 3.811, 4.037],
    'trt': [1.496, 2.947, 0.600, 2.088, 2.098],
    'hidet': [1.329, 1.753, 0.349, 2.551, 2.815]
}

batch_data = {
    'torch': [end2end_data['torch'][0], 7.822, 7.728],
    'ort': [end2end_data['ort'][0], 3.529, 5.322],
    'autotvm': [end2end_data['autotvm'][0], 3.599, 6.054],
    'ansor': [end2end_data['ansor'][0], 3.336, 5.528],
    'trt': [end2end_data['trt'][0], 2.719, 4.321],
    'hidet': [end2end_data['hidet'][0], 3.021, 5.068]
}
batch_sizes = [1, 4, 8]


for name, full_name in exec_fullname.items():
    exec_color[full_name] = exec_color[name]
    exec_edge_color[full_name] = exec_edge_color[name]

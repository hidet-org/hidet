import hidet
from hidet.graph.ops.definitions.normalize import layer_norm

a = hidet.randn([2, 3, 512, 512], device="cpu")
# TODO: rsqrtf doesn't exist in c++ only in cuda so change the compilation
b = layer_norm(a)
# print(b)


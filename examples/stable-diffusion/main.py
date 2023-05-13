import gradio as gr
import torch
from torch import autocast
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

import time
import hidet


model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

dpm = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.half, scheduler=dpm).to("cuda")

unet = pipe.unet.eval()
compiled_unet = torch.compile(unet, backend='hidet')


block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")

num_samples = 1
eager_time = 0
compile_time = 0

def infer(prompt, compile):
    global eager_time, compile_time
    if compile:
        pipe.unet = compiled_unet
    else:
        pipe.unet = unet
    
    with autocast("cuda"):
        start = time.time()
        images = pipe([prompt] * num_samples, guidance_scale=7.5).images
        dur = time.time() - start
    
    if compile:
        compile_time = dur
    else:
        eager_time = dur
    
    time_text = ""
    if eager_time == 0:
        time_text += "eagar time: ..."
    else:
        time_text += f"eagar time: {eager_time}"
    if compile_time == 0:
        time_text += " | compile time: ..."
    else:
        time_text += f" | compile time: {compile_time}"
    
    return images, time_text

def adjust_compile_opts(search_space=0, fp16=True, tensor_core=True, fp_16_red=False, cuda_graph=True, attn=False):
    global compiled_unet
    search_space = max(min(search_space, 2), 0)
    hidet.torch.dynamo_config.search_space(search_space)
    hidet.torch.dynamo_config.use_fp16(fp16)
    hidet.torch.dynamo_config.use_tensor_core(flag=tensor_core)

    compiled_unet = torch.compile(unet, backend='hidet')
    return search_space

adjust_compile_opts()

with block as demo:
    gr.Markdown("<h1><center>Stable Diffusion</center></h1>")
    gr.Markdown(
        "Hidet inferance demo, expect first run under compilation to take longer."
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1, value="A fantasy landscape with a river and mountains distant in the background"
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Run").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
                to_compile = gr.Checkbox(False, label='hidet compile')
            with gr.Accordion("compilation options", open=False):
                with gr.Row():
                    search_space = gr.Number(0, label="search space (0-2, the greater the faster)", elem_classes=[0, 1, 2])
                    fp_16 = gr.Checkbox(True, label="use fp16")
                    tensor_core = gr.Checkbox(True, label='use tensor core')
                with gr.Row():
                    fp_16_red = gr.Checkbox(False, label="use fp16 reduction")
                    cuda_graph = gr.Checkbox(True, label="use cuda graph")
                    attn = gr.Checkbox(False, label="use fused attention schedule")
            gr_time_text = gr.Text("eagar time: ... | compile time: ...", show_label=False)
        
        search_space.change(adjust_compile_opts, inputs=[search_space, fp_16, tensor_core, fp_16_red, cuda_graph, attn], outputs=[search_space])
        fp_16.change(adjust_compile_opts, inputs=[search_space, fp_16, tensor_core, fp_16_red, cuda_graph, attn])
        tensor_core.change(adjust_compile_opts, inputs=[search_space, fp_16, tensor_core, fp_16_red, cuda_graph, attn])
        fp_16_red.change(adjust_compile_opts, inputs=[search_space, fp_16, tensor_core, fp_16_red, cuda_graph, attn])
        cuda_graph.change(adjust_compile_opts, inputs=[search_space, fp_16, tensor_core, fp_16_red, cuda_graph, attn])
        attn.change(adjust_compile_opts, inputs=[search_space, fp_16, tensor_core, fp_16_red, cuda_graph, attn])

        gallery = gr.Gallery(label="Generated images", show_label=False).style(
            grid=[2], height="auto"
        )
    
    text.submit(infer, inputs=[text, to_compile], outputs=[gallery, gr_time_text])
    btn.click(infer, inputs=[text, to_compile], outputs=[gallery, gr_time_text])
    
    gr.Markdown(
        """___
   <p style='text-align: center'>
   CentML: https://github.com/hidet-org/hidet
   <br/>
   </p>"""
    )


demo.launch(debug=True)
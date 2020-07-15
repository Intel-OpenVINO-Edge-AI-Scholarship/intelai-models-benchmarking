import argparse
import os
from .inference import Network
import numpy as np
import cv2

CPU_EXTENSION = "/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/nd131-openvino-people-counter-newui/custom_layers/arcface/cl_pnorm/user_ie_extensions/cpu/build/libpnorm_cpu_extension.so"
VPU_EXTENSION = "/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libmyriadPlugin.so"

def load_to_IE(device, extension, xml_path, factor, model_xml):
    
    network = Network()

    CPU_EXTENSION = extension

    def exec_f(l):
        pass

    network.load_core(model_xml, device, cpu_extension=CPU_EXTENSION)

    if "MYRIAD" in device:
        network.feed_custom_layers(device, {'xml_path': xml_path}, exec_f)

    if "CPU" in device:
        network.feed_custom_parameters(device, exec_f)

    network.load_model(model_xml, device, cpu_extension=CPU_EXTENSION)

    # ### TODO: Load IR files into their related class
    # model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # net = IENetwork(model=model_xml, weights=model_bin)

    return network

def run(network, img, factor, verbose=1):

    img = img.astype(np.float64) - img.min()*factor
    img = img.astype(np.uint8)

    network.sync_inference(img)

    res = network.extract_output()

    print(res)

    return res
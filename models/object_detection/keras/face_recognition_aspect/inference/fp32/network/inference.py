'''
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
'''

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import numpy as np

class Network:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_core(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        # Initialize the plugin
        self.plugin = IECore()

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        return

    def load_model(self, model, device="CPU", cpu_extension=None):
        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

    def feed_custom_layers(self, device, config, callback_function):
        self.plugin.set_config({"VPU_CUSTOM_LAYERS": config['xml_path']}, device)
        # self.plugin.set_initial_affinity(self.network)
        # run the callback_function for the layers identified
        for l in self.network.layers.values():
            callback_function(l)

    def feed_custom_parameters(self, device, callback_function):
        for l in self.network.layers.values():
            callback_function(l)

    def check_layers(self, device):
        supported_layers = self.plugin.query_network(network=self.network, device_name=device)

        ###       know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        self.plugin.load_network(self.network, device)

        print("IR successfully loaded into Inference Engine.")


    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def async_inference(self, image, request_id=0):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_blob: image})
        return

    def sync_inference(self, images):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        images = np.vstack(images)
        self.exec_network.infer({self.input_blob: images})
        return


    def wait(self, request_id=0):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[request_id].wait(-1)
        return status


    def extract_output(self, request_id=0):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[request_id].outputs[self.output_blob]

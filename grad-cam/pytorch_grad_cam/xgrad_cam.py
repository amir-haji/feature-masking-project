import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM


class XGradCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None):
        super(
            XGradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / \
            (sum_activations[:, :, None, None] + eps)
        
        weights = weights.sum(axis=(2, 3))
        return weights

    def get_cam_per_channel(self,
                        input_tensor,
                        target_layer,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = np.abs(grads * activations / \
            (sum_activations[:, :, None, None] + eps))

        return weights

    def get_linear_cam(self,
                        input_tensor,
                        target_layer,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(-1))
        eps = 1e-7
        weights = np.abs(grads * activations / \
            (sum_activations[:, None] + eps))

        return weights

    

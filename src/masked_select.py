from torch import nn 
import torch 

class MaskedImageSelect(nn.Module): 
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, X, mask): 
        B, C, H, W = X.shape
        mask = torch.nn.functional.interpolate(mask.float(), size=(H, W))

        # move channel dim to the end
        X = X.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)

        mask = mask > self.threshold

        outputs = []
        for b in range(B): 
            x = X[b]
            m = mask[b]
            x = x[m]
            outputs.append(x)

        return outputs
    

if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import data

    image = data.camera()
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
    mask = image > 100

    masked_select = MaskedImageSelect()
    outputs = masked_select(image, mask)

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image.squeeze())
    axes[1].imshow(mask.squeeze())
    

    outputs[0][:] = 255 # set all values to 0 inside the mask
    axes[2].imshow(image.squeeze())
    plt.show()
    plt.savefig('masked_select.png')

    # fig, axes = plt.subplots(1, len(outputs))
    # for i, ax in enumerate(axes): 
    #     ax.imshow(outputs[i].squeeze())
    # plt.show()
    # plt.savefig('masked_select_outputs.png')

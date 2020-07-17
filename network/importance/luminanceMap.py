from importance.importanceMap import ImportanceMap

import numpy as np
import array
import torch
from typing import List, Tuple

class LuminanceImportanceMap(ImportanceMap):
    """
    Importance-map based on the code from
    "Luminance-contrast-aware foveated rendering", O.T. Tursun et al., 2019.

    This class calls external Matlab code and is not differentiable.
    """

    MATLAB_INSTANCE = None

    @staticmethod
    def initMatlab(path_to_scripts : str):
        if LuminanceImportanceMap.MATLAB_INSTANCE is not None:
            return # already initialized

        print("Initialize Matlab engine")
        import matlab.engine
        LuminanceImportanceMap.MATLAB_INSTANCE = matlab.engine.start_matlab()
        LuminanceImportanceMap.MATLAB_INSTANCE.cd(path_to_scripts, nargout=0)
        print("Matlab engine started")

    def __init__(self, upsampling_factor:float, 
                 channels:List[int], 
                 range:Tuple[float, float],
                 path_to_matlab_script : str,
                 luminance_upsample : int):
        """
        upsampling_fator: post upsampling factor
        channels: a list of three integers specifying which channels to use
        range: a tuple of two floats (min, max) specifying the value range for the channels
        path_to_matlab_script: where are the files for the luminance matlab code located
        luminance_upsample: upsample images beforehand to get a better prediction
        """
        super().__init__(upsampling_factor)

        assert len(channels)==3
        self._channels = channels
        self._min, self._max = range
        assert luminance_upsample > 0
        self._luminance_upsample = luminance_upsample

        LuminanceImportanceMap.initMatlab(path_to_matlab_script)

    def forward(self, input):
        import matlab
        # to numpy, select channels and normalize
        data = input.cpu().numpy()
        data = data[:, self._channels, :, :]
        data = 255 * (data - self._min) / (self._max - self._min)
        data = data.astype(np.uint8)

        def numpy2matlab(a):
            assert len(a.shape)==3
            assert a.shape[2]==3
            p1 = matlab.uint8(a[:,:,0].tolist())
            p2 = matlab.uint8(a[:,:,1].tolist())
            p3 = matlab.uint8(a[:,:,2].tolist())
            p = LuminanceImportanceMap.MATLAB_INSTANCE.horzcat(p1, p2, p3)
            p = LuminanceImportanceMap.MATLAB_INSTANCE.reshape(p, a.shape[0], a.shape[1], 3)
            #print("converted numpy array", a.shape, "to matlab", p.size)
            return p
        def matlab2numpy(C):
            assert len(C.size)==2
            a = np.array(C._data.tolist())
            a = a.reshape(tuple(reversed(list(C.size)))).transpose()
            #print("converted matlab array", C.size, "to numpy array", a.shape)
            return a

        # loop over batches and compute luminance
        #print("compute luminance of input", data.shape, data.dtype)
        output = []
        for b in range(data.shape[0]):
            batch = data[b, ...].transpose((1,2,0))
            output.append(
                matlab2numpy(
                    LuminanceImportanceMap.MATLAB_INSTANCE.imresize(
                        LuminanceImportanceMap.MATLAB_INSTANCE.run_on_image(
                            LuminanceImportanceMap.MATLAB_INSTANCE.imresize(
                                numpy2matlab(batch), self._luminance_upsample),
                            1.0),
                        1.0/self._luminance_upsample)))
        output = np.stack(output, axis=0)
        #print("Done, output:", output.shape, output.dtype)

        # flip it and convert back to torch
        output = 1 - output
        output = torch.from_numpy(output).to(dtype=input.dtype, device=input.device)
        return self._upsample(output).unsqueeze(1)


def __importanceFromDset():
    """
    computes the importance map for a whole dataset
    """
    from contextlib import ExitStack
    import h5py
    from console_progressbar import ProgressBar

    DATASET_PREFIX = "D:/VolumeSuperResolution-InputData/"
    DSET = "gt-rendering-ejecta-v2-test.hdf5"
    OUTPUT = "gt-rendering-ejecta-v2-test-luminanceImportance.hdf5"

    with ExitStack() as stack:
        inputFile = stack.enter_context(h5py.File(DATASET_PREFIX+DSET, 'r'))
        outputFile = stack.enter_context(h5py.File(DATASET_PREFIX+OUTPUT, 'w'))

        inputDset = inputFile['gt']
        B, T, C, H, W = inputDset.shape
        outputDsetShape = (B, T, 1, H, W)
        outputDset = outputFile.create_dataset(
            'importance',
            outputDsetShape,
            dtype=inputDset.dtype)

        importanceMap = LuminanceImportanceMap(
            1, [0,1,2], (0,1),
            "..\\..\\tests\\luminance-contrast\\siggraph2019-matlab", 
            2)

        pg = ProgressBar(B*T, 'Importance', length=50)
        for b in range(B):
            for t in range(T):
                pg.print_progress_bar(t + T*b)

                input = inputDset[b, t, :, :, :]
                input_torch = torch.from_numpy(input).unsqueeze(0)
                output_torch = importanceMap(input_torch)
                outputDset[b, t, :, :, :] = output_torch.numpy()

        pg.print_progress_bar(B*T)
        print("Done")

def __importanceFromImage():
    import imageio
    INPUT_IMAGE = "../result-stats/adaptiveIsoEnhance6/importanceInput.png"
    OUTPUT_IMAGE = "../result-stats/adaptiveIsoEnhance6/importanceLuminance.png"

    img = imageio.imread(INPUT_IMAGE, pilmode='RGB') / 255.0
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
    img = img.to(dtype=torch.float32)
    #img = torch.nn.functional.interpolate(img, scale_factor=1/4, mode='bilinear')

    importanceMap = LuminanceImportanceMap(
            1, [0,1,2], (0,1),
            "..\\..\\tests\\luminance-contrast\\siggraph2019-matlab", 
            2)

    output = importanceMap(img)
    print(output.shape)
    #output = torch.nn.functional.interpolate(output, scale_factor=4, mode='bilinear')

    img = np.clip(output[0,0].detach().cpu().numpy(), 0, 1)
    img = np.stack([img]*3, axis=2)
    imageio.imwrite(OUTPUT_IMAGE, img)

if __name__ == "__main__":
    #__importanceFromDset()
    __importanceFromImage()
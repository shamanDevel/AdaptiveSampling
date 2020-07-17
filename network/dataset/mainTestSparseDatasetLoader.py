import matplotlib.pyplot as plt
import datetime
import numpy as np

from sparseDatasetLoader import SparseDatasetLocator, SparseDataset

if __name__ == "__main__":
    root = "D:/VolumeSuperResolution-InputData/sparse-rendering"

    locator = SparseDatasetLocator([root], 0.2)
    training_samples = locator.get_training_samples()
    print("Training samples:")
    for s in training_samples:
        print(" ", s[1])
    test_samples = locator.get_test_samples();
    print("Test samples:")
    for s in test_samples:
        print(" ", s[1])

    print()

    # samples for visualization
    vis_samples = training_samples[0:5]

    # full resolution
    fullres_dataset = SparseDataset(
        samples=vis_samples, 
        crops_per_sample=None,
        buffer_size=2)
    print(datetime.datetime.now().time(), "Full Resolution, num entries: ",
          fullres_dataset.length())
    fig, ax = plt.subplots(fullres_dataset.length(), 1)
    for idx, (sparse, dense) in enumerate(fullres_dataset):
        print(datetime.datetime.now().time(),"% 4d"%idx,dense.shape)
        ax[idx].imshow(dense[0,1:4,:,:].transpose((1,2,0))*0.5+0.5)
    fig.suptitle("Full Resolution")
    #fig.show()

    # crops
    crop_dataset = SparseDataset(
        samples=vis_samples, 
        crops_per_sample=10,
        crop_size=64,
        buffer_size=2)
    print(datetime.datetime.now().time(), "Crops, num entries: ",
          crop_dataset.length())
    fig, ax = plt.subplots(5, 10)
    for idx, (sparse, dense) in enumerate(crop_dataset):
        mask01 = dense[0,0,:,:]*0.5+0.5
        fill_rate = np.sum(np.abs(mask01)) / (dense.shape[2]*dense.shape[3])
        print(datetime.datetime.now().time(),"% 4d"%idx,dense.shape," fill-rate=%5.3f"%fill_rate)
        ax[idx//10, idx%10].imshow(dense[0,1:4,:,:].transpose((1,2,0))*0.5+0.5)
        #ax[idx//10, idx%10].imshow(dense[0,0,:,:]*0.5+0.5)
    fig.suptitle("Crops")
    #fig.show()

    plt.show()
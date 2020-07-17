import numpy as np
from math import sqrt, exp


# helper for loop index calculations
def start_index(a, b, c):
    return int(np.floor(a*c/b))
def end_index(a, b, c):
    return int(np.ceil((a+1)*c/b))



def downsampleForward(mask, data):
    C, H, W = data.shape
    oH, oW = H//2, W//2
    
    maskLow = np.zeros((oH, oW), dtype=mask.dtype)
    dataLow = np.zeros((C, oH, oW), dtype=data.dtype)
    for i in range(oH):
        for j in range(oW):
            N1 = 0
            N2 = 0
            Count = 0
            d = np.zeros(C, dtype=data.dtype)
            for ii in range(start_index(i, oH, H), end_index(i, oH, H)):
                for jj in range(start_index(j, oW, W), end_index(j, oW, W)):
                    N1 += mask[ii,jj]
                    N2 = max(N2, mask[ii,jj])
                    d += mask[ii,jj] * data[:,ii,jj]
                    Count += 1
            if N1>0:
                maskLow[i,j] = N2
                dataLow[:,i,j] = d / N1
                
    return maskLow, dataLow

def upsampleForward(mask, data, maskLow, dataLow):
    C, H, W = data.shape
    oH, oW = H//2, W//2
    
    maskHigh = np.zeros((H, W), dtype=mask.dtype)
    dataHigh = np.zeros((C, H, W), dtype=data.dtype)
    for i in range(H):
        for j in range(W):
            # interpolate from low resolution (bilinear)
            # get neighbors with weight
            nx = []
            nx += [(i//2, j//2, 0.75*0.75)]
            io = -1 if i%2==0 else +1
            jo = -1 if j%2==0 else +1
            nx += [(i//2+io, j//2, 0.25*0.75), (i//2, j//2+jo, 0.25*0.75)]
            nx += [(i//2+io, j//2+jo, 0.25*0.25)]
            # accumulate
            N = 0
            d = np.zeros(C, dtype=data.dtype)
            WW = 0
            for (ii,jj,w) in nx:
                if ii>=0 and jj>=0 and ii<oH and jj<oW:
                    N += w * maskLow[ii,jj]
                    d += w * maskLow[ii,jj] * dataLow[:,ii,jj]
                    WW += w
            # blend with original data
            if N>0:
                maskHigh[i,j] = mask[i,j] + (1-mask[i,j]) * N / WW
                dataHigh[:,i,j] = mask[i,j] * data[:,i,j] + (1-mask[i,j]) * d / N
            else:
                maskHigh[i,j] = mask[i,j]
                dataHigh[:,i,j] = mask[i,j] * data[:,i,j]
                
    return maskHigh, dataHigh

# fractional inpainting (forward)
def inpaintFractional(mask, data, iStack = None):
    """
    mask: (H, W)
    data: (C, H, W)
    callback: function with callback(level, maskPre, dataPre, maskPost, dataPost) (can be None)
    """
    
    C, H, W = data.shape
    oH, oW = H//2, W//2
    # end of recursion
    if H<=1 or W<=1:
        return mask, data
    
    # downsample
    maskLowPre, dataLowPre = downsampleForward(mask, data)
    
    # recursion
    maskLowPost, dataLowPost = inpaintFractional(maskLowPre, dataLowPre, iStack)
    
    # upsample
    maskHigh, dataHigh = upsampleForward(mask, data, maskLowPost, dataLowPost)
                
    # save for adjoint
    if iStack is not None:
        iStack.append(maskLowPre)
        iStack.append(dataLowPre)
        iStack.append(maskLowPost)
        iStack.append(dataLowPost)
                
    return maskHigh, dataHigh



def upsampleBackward(
        maskIn, dataIn, # the original data before the downsampling+recursion
        maskLowIn, dataLowIn, # the downsampled data after recursion
        adjMaskHighIn, adjDataHighIn # gradient of the output high-resolution mask+data
        ):
    # Returns: 
    #  adjMask, adjData: gradients of the original (high-res) data before downsampling+recursion
    #  adjMaskLow, adjDataLow: gradients of the downsampled data after recursion
    
    C, H, W = dataIn.shape
    oH, oW = H//2, W//2
    
    adjMask = np.zeros((H, W), dtype=maskIn.dtype)
    adjData = np.zeros((C, H, W), dtype=dataIn.dtype)
    adjMaskLow = np.zeros((oH, oW), dtype=maskIn.dtype)
    adjDataLow = np.zeros((C, oH, oW), dtype=dataIn.dtype)
    for i in range(H):
        for j in range(W):
            # get neighbors with weight
            nx = []
            nx += [(i//2, j//2, 0.75*0.75)]
            io = -1 if i%2==0 else +1
            jo = -1 if j%2==0 else +1
            nx += [(i//2+io, j//2, 0.25*0.75), (i//2, j//2+jo, 0.25*0.75)]
            nx += [(i//2+io, j//2+jo, 0.25*0.25)]
            
            # run forward again
            N = 0
            d = np.zeros(C, dtype=dataIn.dtype)
            WW = 0
            for (ii,jj,w) in nx:
                if ii>=0 and jj>=0 and ii<oH and jj<oW:
                    N += w * maskLowIn[ii,jj]
                    d += w * maskLowIn[ii,jj] * dataLowIn[:,ii,jj]
                    WW += w
                    
            # adjoint of output
            m = maskIn[i,j]
            adjD = np.zeros(C, dtype=dataIn.dtype)
            adjN = 0
            adjWW = 0
            # dataHigh[:,i,j] = mask[i,j] * data[:,i,j] + (1-mask[i,j]) * d / N
            adjMask[i,j] += np.dot(adjDataHighIn[:,i,j], dataIn[:,i,j])
            adjData[:,i,j] += adjDataHighIn[:,i,j] * m
            if N>0:
                adjMask[i,j] -= np.dot(adjDataHighIn[:,i,j], d / N)
                adjD += adjDataHighIn[:,i,j] * (1-m) / N
                adjN -= np.dot(adjDataHighIn[:,i,j], (1-m)*d/(N*N))
            # maskHigh[i,j] = mask[i,j] + (1-mask[i,j]) * N / WW
            adjMask[i,j] += adjMaskHighIn[i,j]
            if N>0:
                adjMask[i,j] -= adjMaskHighIn[i,j] * N / WW
                adjN += adjMaskHighIn[i,j] * (1-m) / WW
                adjWW -= adjMaskHighIn[i,j] * (1-m) / (WW*WW) #not needed
                
            # adjoint of accumulation
            for (ii,jj,w) in nx:
                if ii>=0 and jj>=0 and ii<oH and jj<oW:
                    # d += w * maskLow[ii,jj] * dataLow[:,ii,jj]
                    adjMaskLow[ii,jj] += np.dot(adjD, w * dataLowIn[:,ii,jj])
                    adjDataLow[:,ii,jj] += adjD * w * maskLowIn[ii,jj]
                    # N += w * maskLow[ii,jj]
                    adjMaskLow[ii,jj] += adjN * w
                    
    return adjMask, adjData, adjMaskLow, adjDataLow

def downsampleBackward(
        maskIn, dataIn, # high-resolution input before downsampling+recursion
        adjMaskLowIn, adjDataLowIn # gradients of low-resolution output
        ):
    # Returns:
    #  adjMask, adjData: gradients of the high-resolution input
    
    C, H, W = dataIn.shape
    oH, oW = H//2, W//2
    
    adjMask = np.zeros((H, W), dtype=maskIn.dtype)
    adjData = np.zeros((C, H, W), dtype=dataIn.dtype)
    
    for i in range(oH):
        for j in range(oW):
            # run forward again
            N1 = 0
            N2 = 0
            Count = 0
            d = np.zeros(C, dtype=dataIn.dtype)
            for ii in range(start_index(i, oH, H), end_index(i, oH, H)):
                for jj in range(start_index(j, oW, W), end_index(j, oW, W)):
                    N1 += maskIn[ii,jj]
                    N2 = max(N2, maskIn[ii,jj])
                    d += maskIn[ii,jj] * dataIn[:,ii,jj]
                    Count += 1
                    
            # adjoint: output
            adjD = np.zeros_like(d)
            adjN1 = 0
            adjN2 = 0
            if N1>0:
                # dataLow[:,i,j] = d / N1
                adjD += adjDataLowIn[:,i,j] / N1
                adjN1 -= np.dot(adjDataLowIn[:,i,j], d) / (N1*N1)
                # maskLow[i,j] = N2
                adjN2 += adjMaskLowIn[i,j]
            # adjoint interpolation
            for ii in range(start_index(i, oH, H), end_index(i, oH, H)):
                for jj in range(start_index(j, oW, W), end_index(j, oW, W)):
                    # d += maskIn[ii,jj] * dataIn[:,ii,jj]
                    adjMask[ii,jj] += np.dot(adjD, dataIn[:,ii,jj])
                    adjData[:,ii,jj] += adjD * maskIn[ii,jj]
                    # N2 = max(N2, maskIn[ii,jj])
                    if N2==maskIn[ii,jj]:
                        adjMask[ii,jj] += adjN2
                    # N1 += maskIn[ii,jj]
                    adjMask[ii,jj] += adjN1

    return adjMask, adjData
                    
# fraction inpainting, adjoint code
def adjInpaintFractional_recursion(maskIn, dataIn, gradMaskIn, gradDataIn, iStack):
    
    C, H, W = dataIn.shape
    oH, oW = H//2, W//2
    #print("H:",H,", W:",W,", oH:",oH, ", oW:",oW)
    
    gradMaskOut = np.zeros((H, W), dtype=maskIn.dtype)
    gradDataOut = np.zeros((C, H, W), dtype=dataIn.dtype)
    
    # end of recursion
    if H<=1 or W<=1:
        assert gradDataOut.shape == gradDataIn.shape
        gradMaskOut += gradMaskIn
        gradDataOut += gradDataIn
        return gradMaskOut, gradDataOut
        
    # get saved tensors (after recursion)
    dataLowPost = iStack.pop()
    maskLowPost = iStack.pop()
    dataLowPre = iStack.pop()
    maskLowPre = iStack.pop()
    
    # adjoint upsample
    adjMask, adjData, adjMaskLowPost, adjDataLowPost = upsampleBackward(
        maskIn, dataIn, maskLowPost, dataLowPost, gradMaskIn, gradDataIn)
    gradMaskOut += adjMask
    gradDataOut += adjData
                    
    # recursion
    adjMaskLowPre, adjDataLowPre = adjInpaintFractional_recursion(
        maskLowPre, dataLowPre, adjMaskLowPost, adjDataLowPost, 
        iStack)
    
    # adjoint downsample
    adjMask, adjData = downsampleBackward(
        maskIn, dataIn, adjMaskLowPre, adjDataLowPre)
    gradMaskOut += adjMask
    gradDataOut += adjData
    
    return gradMaskOut, gradDataOut

def adjInpaintFractional(mask, data, gradOutput):
    # run forward again, but save results
    s = []
    inpaintFractional(mask, data, s)
    print("stack size:", len(s))
    
    # adjoint recursion
    gradMask = np.zeros_like(mask)
    outGradMask, outGradData = adjInpaintFractional_recursion(
    mask, data, gradMask, gradOutput,  s)
    
    assert len(s)==0
    
    return outGradMask, outGradData




# testing code
def validateFull(size, channels=1, epsilon=1e-5):
    print()
    print("Validate full algorithm with size", size)
    # create inputs
    maskIn = np.random.rand(size, size)
    dataIn = np.random.rand(channels, size, size)
    gradDataOut = np.random.rand(channels, size, size)
    print("Mask:\n", maskIn)
    print("Data:\n", dataIn)
    print("adjResult:\n", gradDataOut)
    
    # forward computation (for testing)
    maskOut, dataOut = inpaintFractional(maskIn, dataIn)
    print()
    print("Forward computation:")
    print("Mask-Out:\n", maskOut)
    print("Data-Out:\n", dataOut)
    
    # analytic gradient
    anaGradMask, anaGradData = adjInpaintFractional(maskIn, dataIn, gradDataOut)
    
    # numeric
    print("\nEpsilon:", epsilon)
    
    # mask
    print("\nCompare against numeric gradients for mask")
    for x in range(size):
        for y in range(size):
            # mask
            maskIn2 = maskIn.copy()
            maskIn2[x,y] += epsilon
            _, dataOut2 = inpaintFractional(maskIn2, dataIn)
            numGrad = np.dot((dataOut2 - dataOut).reshape((channels*size*size)), 
                             gradDataOut.reshape((channels*size*size))) / epsilon
            anaGrad = anaGradMask[x,y]
            print("[%2d, %2d] analytic=%7.5f, numeric=%7.5f"%(x,y,anaGrad,numGrad))
            
    # data
    print("\nCompare against numeric gradients for data")
    for x in range(size):
        for y in range(size):
            for c in range(channels):
                # data
                dataIn2 = dataIn.copy()
                dataIn2[c,x,y] += epsilon
                _, dataOut2 = inpaintFractional(maskIn, dataIn2)
                numGrad = np.dot((dataOut2 - dataOut).reshape((channels*size*size)), 
                                 gradDataOut.reshape((channels*size*size))) / epsilon
                anaGrad = anaGradData[c,x,y]
                print("[%2d, %2d, c=%d] analytic=%7.5f, numeric=%7.5f"%(x,y,c,anaGrad,numGrad))

# partial validation
def validateDownsample(size, channels=1, epsilon=1e-5):
    print()
    print("Validate full algorithm with size", size)
    oSize = size//2
    # create inputs
    maskIn = np.random.rand(size, size)
    dataIn = np.random.rand(channels, size, size)
    adjMaskLow = np.random.rand(oSize, oSize)
    adjDataLow = np.random.rand(channels, oSize, oSize)
    print("Mask:\n", maskIn)
    print("Data:\n", dataIn)
    print("adjMaskLow:\n", adjMaskLow)
    print("adjDataLow:\n", adjDataLow)
    
    # forward computation (for testing)
    maskLow, dataLow = downsampleForward(maskIn, dataIn)
    print()
    print("Forward computation:")
    print("Mask-Out:\n", maskLow)
    print("Data-Out:\n", dataLow)
    
    # analytic gradient
    anaAdjMask, anaAdjData = downsampleBackward(maskIn, dataIn, adjMaskLow, adjDataLow)
    
    # numeric
    print("\nEpsilon:", epsilon)
    
    # mask
    print("\nCompare against numeric gradients for mask")
    for x in range(size):
        for y in range(size):
            # mask
            maskIn2 = maskIn.copy()
            maskIn2[x,y] += epsilon
            maskLow2, dataLow2 = downsampleForward(maskIn2, dataIn)
            numGrad = np.dot((maskLow2 - maskLow).reshape((oSize*oSize)), 
                             adjMaskLow.reshape((oSize*oSize))) / epsilon
            numGrad += np.dot((dataLow2 - dataLow).reshape((channels*oSize*oSize)), 
                             adjDataLow.reshape((channels*oSize*oSize))) / epsilon
            anaGrad = anaAdjMask[x,y]
            print("[%2d, %2d] analytic=%7.5f, numeric=%7.5f"%(x,y,anaGrad,numGrad))
            
    # data
    print("\nCompare against numeric gradients for data")
    for x in range(size):
        for y in range(size):
            for c in range(channels):
                # data
                dataIn2 = dataIn.copy()
                dataIn2[c,x,y] += epsilon
                maskLow2, dataLow2 = downsampleForward(maskIn, dataIn2)
                numGrad = np.dot((maskLow2 - maskLow).reshape((oSize*oSize)), 
                             adjMaskLow.reshape((oSize*oSize))) / epsilon
                numGrad += np.dot((dataLow2 - dataLow).reshape((channels*oSize*oSize)), 
                                 adjDataLow.reshape((channels*oSize*oSize))) / epsilon
                anaGrad = anaAdjData[c,x,y]
                print("[%2d, %2d, c=%d] analytic=%7.5f, numeric=%7.5f"%(x,y,c,anaGrad,numGrad))

def validateUpsample(size, channels=1, epsilon=1e-10):
    print()
    print("Validate full algorithm with size", size)
    oSize = size//2
    # create inputs
    mask = np.random.rand(size, size)
    data = np.random.rand(channels, size, size)
    maskLow = np.random.rand(oSize, oSize)
    dataLow = np.random.rand(channels, oSize, oSize)
    adjMaskHigh = np.random.rand(size, size)
    adjDataHigh = np.random.rand(channels, size, size)
    print("Mask:\n", mask)
    print("Data:\n", data)
    print("MaskLow:\n", maskLow)
    print("DataLow:\n", dataLow)
    print("adjMaskHigh:\n", adjMaskHigh)
    print("adjDataHigh:\n", adjDataHigh)
    
    # forward computation (for testing)
    maskHigh, dataHigh = upsampleForward(mask, data, maskLow, dataLow)
    print()
    print("Forward computation:")
    print("MaskHigh:\n", maskHigh)
    print("DataHigh:\n", dataHigh)
    
    # analytic gradient
    anaAdjMask, anaAdjData, anaAdjMaskLow, anaAdjDataLow = \
        upsampleBackward(mask, data, maskLow, dataLow, adjMaskHigh, adjDataHigh)
    
    # numeric
    print("\nEpsilon:", epsilon)
    
    # mask high
    print("\nCompare against numeric gradients for maskHigh")
    for x in range(size):
        for y in range(size):
            mask2 = mask.copy()
            mask2[x,y] += epsilon
            maskHigh2, dataHigh2 = upsampleForward(mask2, data, maskLow, dataLow)
            numGrad = np.dot((maskHigh2 - maskHigh).reshape((size*size)), 
                             adjMaskHigh.reshape((size*size))) / epsilon
            numGrad += np.dot((dataHigh2 - dataHigh).reshape((channels*size*size)), 
                             adjDataHigh.reshape((channels*size*size))) / epsilon
            anaGrad = anaAdjMask[x,y]
            print("[%2d, %2d] analytic=%7.5f, numeric=%7.5f"%(x,y,anaGrad,numGrad))
    
    # data high
    print("\nCompare against numeric gradients for dataHigh")
    for x in range(size):
        for y in range(size):
            for c in range(channels):
                data2 = data.copy()
                data2[c,x,y] += epsilon
                maskHigh2, dataHigh2 = upsampleForward(mask, data2, maskLow, dataLow)
                numGrad = np.dot((maskHigh2 - maskHigh).reshape((size*size)), 
                                 adjMaskHigh.reshape((size*size))) / epsilon
                numGrad += np.dot((dataHigh2 - dataHigh).reshape((channels*size*size)), 
                                 adjDataHigh.reshape((channels*size*size))) / epsilon
                anaGrad = anaAdjData[c,x,y]
                print("[%2d, %2d, c=%d] analytic=%7.5f, numeric=%7.5f"%(x,y,c,anaGrad,numGrad))
    
    # mask low
    print("\nCompare against numeric gradients for maskLow")
    for x in range(oSize):
        for y in range(oSize):
            maskLow2 = maskLow.copy()
            maskLow2[x,y] += epsilon
            maskHigh2, dataHigh2 = upsampleForward(mask, data, maskLow2, dataLow)
            numGrad = np.dot((maskHigh2 - maskHigh).reshape((size*size)), 
                             adjMaskHigh.reshape((size*size))) / epsilon
            numGrad += np.dot((dataHigh2 - dataHigh).reshape((channels*size*size)), 
                             adjDataHigh.reshape((channels*size*size))) / epsilon
            anaGrad = anaAdjMaskLow[x,y]
            print("[%2d, %2d] analytic=%7.5f, numeric=%7.5f"%(x,y,anaGrad,numGrad))
            
    # data high
    print("\nCompare against numeric gradients for dataLow")
    for x in range(oSize):
        for y in range(oSize):
            for c in range(channels):
                dataLow2 = dataLow.copy()
                dataLow2[c,x,y] += epsilon
                maskHigh2, dataHigh2 = upsampleForward(mask, data, maskLow, dataLow2)
                numGrad = np.dot((maskHigh2 - maskHigh).reshape((size*size)), 
                                 adjMaskHigh.reshape((size*size))) / epsilon
                numGrad += np.dot((dataHigh2 - dataHigh).reshape((channels*size*size)), 
                                 adjDataHigh.reshape((channels*size*size))) / epsilon
                anaGrad = anaAdjDataLow[c,x,y]
                print("[%2d, %2d, c=%d] analytic=%7.5f, numeric=%7.5f"%(x,y,c,anaGrad,numGrad))





# Execution of test scripts
if __name__ == "__main__":
    validateFull(1)
    validateFull(2)
    validateFull(4)
    validateFull(8)
    
    #validateUpsample(4)
    #validateDownsample(4)
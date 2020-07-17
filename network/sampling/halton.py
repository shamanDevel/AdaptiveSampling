import sys
import numpy as np
import matplotlib.pyplot as plt

class HaltonSampler:
    def __init__(self):
        self.primes = [2,3,5,7,11,13,17,19,23]
        self.radicalInversePermutations = [None] * 24
        for i in self.primes:
            self.radicalInversePermutations[i] = np.arange(i)
            np.random.shuffle(self.radicalInversePermutations[i])

    def radicalInverse(self, a : int, base : int):
        invBase = 1.0 / base
        reversedDigits = 0
        invBaseN = 1.0
        perm = self.radicalInversePermutations[base]
        while a>0:
            next = a // base
            digit = a - next * base
            reversedDigits = reversedDigits * base + perm[digit]
            invBaseN *= invBase;
            a = next
        return min(invBaseN * (reversedDigits + \
            invBase * perm[0] / (1-invBase)), 1 - sys.float_info.epsilon)

    def sample(self, i:int, bases):
        return [self.radicalInverse(i, b) for b in bases]

    def sampleN2D(self, b1:int, b2:int, N:int):
        """
        Returns N random points in [0,1]^2 as a Nx2 array
        """

        a = np.empty((N, 2))

        for i in range(N):
            a[i,0] = self.radicalInverse(i, b1)
            a[i,1] = self.radicalInverse(i, b2)

        return a

    def fill_image(self, shape):
        """
        Creates a d-dimensional integer tensor of shape 'shape'
        in which every entry specifies the index when the 
        sequence visits that entry.
        """
        bases = self.primes[:len(shape)]
        a = np.zeros(shape, dtype=int)
        samples = 0
        target_samples = np.prod(list(shape))
        shape = np.array(list(shape))
        index = 0
        attempts = 0
        while samples<target_samples:
            z = self.sample(index, bases)
            index += 1
            c = np.floor(shape * z).astype(int)
            if a[tuple(c)]==0:
                samples += 1
                a[tuple(c)] = samples
            attempts += 1
        print("Filling", target_samples, "entries took",attempts,'trials')
        return a

def test_simple():
    halton = HaltonSampler()

    print("radical inverse:")
    for b in [2,3,5,7,11,13,17,19]:
        print("b=%02d: "%b, end='')
        for k in range(1,10):
            print(" %6.4f"%halton.radicalInverse(k,b), end='')
        print()


    width = 1920
    height = 1080

    def plot(points, color=True, filename=None):
        if filename is None:
            plt.figure()
        else:
            aspect = height / width
            plt.figure(figsize=(20/aspect,20))
        ax1 = plt.subplot(1, 1, 1)
        if color:
            ax1.scatter(points[:,1]*width, height-points[:,0]*height-1, c=np.arange(points.shape[0]), s=1)
        else:
            ax1.plot(points[:,1]*width, height-points[:,0]*height-1, 'k.')
        ax1.set_xlim(0, width)
        ax1.set_ylim(0, height)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    print("Generate points")
    a = halton.sampleN2D(2, 3, 1000000)
    print("Dohne")
    a_with_time = np.concatenate([a, np.linspace(0, 1, a.shape[0]).reshape((a.shape[0],1))], axis=1)
    #plot(a)

    # Filter it
    #condition = np.sqrt((a_with_time[:,0]-0.5)**2 + (a_with_time[:,1]-0.5)**2)*np.sqrt(2) <= a_with_time[:,2]
    #a2 = a[condition,:]
    #plot(a2)

    # Filter it
    from example_density import density_from_image_grad
    d = density_from_image_grad("NormalEjecta.png")
    d = (d-d.min()) / (d.max()-d.min()) + 0.01
    condition = np.array([d[int(p[0]*(d.shape[0]-1)), int(p[1]*(d.shape[1]-1))] >= p[2] for p in a_with_time])
    a2 = a[condition,:]
    print("Number of points:", a2.shape[0])
    plot(a2, color=False, filename="Ejecta2Halton.png")

def test_filled_image():
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider
    
    W = 200
    H = 200
    sampler = HaltonSampler()
    img = sampler.fill_image((W, H))

    # https://stackoverflow.com/a/46328223/1786598

    f = plt.matshow(img)
    fig = plt.gcf();

    # slider
    axslider = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axslider, 'Amp', 0, 1, valinit=1)

    def update_slider(val):
        global is_manual
        is_manual=True
        update(val)

    def update(val):
        # update curve
        img2 = img.copy()
        img2[img2>val*W*H] = 0
        f.set_data(img2)
        # redraw canvas while idle
        fig.canvas.draw_idle()

    samp.on_changed(update_slider)
    plt.title("Halton Sampling")
    plt.show()

def __write_mapping():
    W = 2048
    H = 2048
    sampler = HaltonSampler()
    print("Fill image")
    img = sampler.fill_image((W, H))
    # normalize
    img = (img.astype(np.float32) / (W * H)).astype(np.float32)
    print("min:", np.min(img), ", max:", np.max(img), ", mean:", np.mean(img), ", dtype:", img.dtype)
    # save
    import os
    filename = os.path.abspath("sampling_halton_2048.bin")
    with open(filename, "wb") as f:
        size = np.array([W, H], dtype=np.int32)
        f.write(size.tobytes('C'))
        f.write(img.tobytes('C'))
    print("Saved to", filename)

if __name__ == "__main__":
    #test_simple()
    #test_filled_image()
    __write_mapping()
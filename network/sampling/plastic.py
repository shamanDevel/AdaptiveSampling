import numpy as np
import math

class PlasticSampler:
    """
    Another low-discrepancy sampler based on
    https://stats.stackexchange.com/questions/25528/do-low-discrepancy-sequences-work-in-discrete-spaces
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """

    def __init__(self, d : int):
        """
        d: number of dimensions
        """
        self._d = d

        def gamma(d): # Use Newton-Rhapson-Method
            x=1.0000
            for i in range(20):
                x = x-(pow(x,d+1)-x-1)/((d+1)*pow(x,d)-1)
            return x
        g = gamma(d)
        self._alpha = np.zeros(d)
        for j in range(d):
            self._alpha[j] = math.pow(1/g,j+1) % 1

    def sample(self, i : int):
        z = (0.5 + self._alpha*(i+1)) % 1
        return z

    def fill_image(self, shape):
        """
        Creates a d-dimensional integer tensor of shape 'shape'
        in which every entry specifies the index when the 
        sequence visits that entry.
        """
        assert len(shape)==self._d, "dimensions must agree"
        a = np.zeros(shape, dtype=int)
        samples = 0
        target_samples = np.prod(list(shape))
        shape = np.array(list(shape))
        index = 0
        attempts = 0
        while samples<target_samples:
            z = self.sample(index)
            index += 1
            c = np.floor(shape * z).astype(int)
            if a[tuple(c)]==0:
                samples += 1
                a[tuple(c)] = samples
            attempts += 1
        print("Filling", target_samples, "entries took",attempts,'trials')
        return a

def __test_gui():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider
    
    W = 32
    H = 32
    sampler = PlasticSampler(2)
    img = sampler.fill_image((W, H))

    # https://stackoverflow.com/a/46328223/1786598

    f = plt.matshow(img)
    fig = plt.gcf();

    # slider
    axslider = plt.axes([0.25, .03, 0.50, 0.02])
    samp = Slider(axslider, 'Amp', 0, 1, valinit=1)

    # animation controls
    is_manual = False # True if user has taken control of the animation
    interval = 100 # ms, time between animation frames
    loop_len = 5.0 # seconds per loop
    scale = interval / 1000 / loop_len

    def update_slider(val):
        global is_manual
        is_manual=True
        update(val)

    def update(val):
        # update curve
        img2 = img.copy()
        img2[img2>val*W*H] = 0
        f.set_data(img2)
        #plt.matshow(img2, fignum=0)
        # redraw canvas while idle
        fig.canvas.draw_idle()

    def update_plot(num):
        global is_manual
        if is_manual:
            return l, # don't change

        val = (samp.val + scale) % 1
        samp.set_val(val)
        is_manual = False # the above line called update_slider, so we need to reset this

    def on_click(event):
        # Check where the click happened
        (xm,ym),(xM,yM) = samp.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            global is_manual
            is_manual=False

    samp.on_changed(update_slider)
    #fig.canvas.mpl_connect('button_press_event', on_click)
    #ani = animation.FuncAnimation(fig, update_plot, interval=interval)
    plt.title("Plastic Sampling")
    plt.show()

def __write_mapping():
    W = 2048
    H = 2048
    sampler = PlasticSampler(2)
    print("Fill image")
    img = sampler.fill_image((W, H))
    # normalize
    img = (img.astype(np.float32) / (W * H)).astype(np.float32)
    print("min:", np.min(img), ", max:", np.max(img), ", mean:", np.mean(img), ", dtype:", img.dtype)
    # save
    import os
    filename = os.path.abspath("sampling_plastic_2048.bin")
    with open(filename, "wb") as f:
        size = np.array([W, H], dtype=np.int32)
        f.write(size.tobytes('C'))
        f.write(img.tobytes('C'))
    print("Saved to", filename)

if __name__ == "__main__":
    #__test_gui()
    __write_mapping()

import numpy as np
import math

class RegularSampler:
    """
    Completely uniform sampling
    """

    def __init__(self, d : int):
        """
        d: number of dimensions
        """
        self._d = d

    def fill_image(self, shape):
        """
        Creates a d-dimensional integer tensor of shape 'shape'
        in which every entry specifies the index when the 
        sequence visits that entry.
        """
        assert len(shape)==self._d, "dimensions must agree"
        log_shape = [int(math.log2(s)) for s in shape]
        assert [2**s for s in log_shape] == list(shape), "shape is not a power of two"

        a = np.zeros(shape, dtype=int)
        samples = 0
        for i in range(log_shape[0]):
            step = 2**(log_shape[0]-i-1)
            for x in range(step-1, shape[0], step):
                for y in range(step-1, shape[0], step):
                    if a[x,y]==0:
                        samples += 1
                        a[x,y] = samples
        return a


def __test_gui():
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.widgets import Slider
    
    W = 256
    H = 256
    sampler = RegularSampler(2)
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
    plt.title("Uniform Sampling")
    plt.show()

def __write_mapping():
    W = 2048
    H = 2048
    sampler = RegularSampler(2)
    print("Fill image")
    img = sampler.fill_image((W, H))
    # normalize
    img = (img.astype(np.float32) / (W * H)).astype(np.float32)
    print("min:", np.min(img), ", max:", np.max(img), ", mean:", np.mean(img), ", dtype:", img.dtype)
    # save
    import os
    filename = os.path.abspath("sampling_uniform_2048.bin")
    with open(filename, "wb") as f:
        size = np.array([W, H], dtype=np.int32)
        f.write(size.tobytes('C'))
        f.write(img.tobytes('C'))
    print("Saved to", filename)

if __name__ == "__main__":
    #__test_gui()
    __write_mapping()

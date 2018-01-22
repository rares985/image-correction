from pathlib import Path
import cv2
import numpy as np

def gradient(x, axis):
    """
        Computes x's gradient along an axis.
    """
    return np.roll(x, -1, axis=axis) - x

def divergence(x, axis):
    """
        Compute x's divergnce along an axis
    """
    return x - np.roll(x, 1, axis=axis)

class Image(object):

    def __init__(self):
        """
            Class initializer.
        """
        self.img = None
        self.grayscaled = None
        self.height = 0
        self.width = 0
        self.channel_count = 0

    def open(self, name):
        """
            Opens an image with the name specified.
            Checks if the file exists before opening,
            and throws an error.
        """
        path = Path(name)
        if not path.is_file():
            raise IOError("Could not open image!")

        self.img = cv2.imread(name, cv2.IMREAD_COLOR)
        self.grayscaled = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.height, self.width, self.channel_count = self.img.shape

    def correct_exposure(self):
        """
            Corrects over/under exposure in images by using
            histogram equalization.
        """
        histogram, bins = np.histogram(self.grayscaled.ravel(), 256, [0, 256])

        # Cumulative sums
        cumsum = histogram.cumsum()

        # Normalize cumulative sums
        cumsum_norm = cumsum * histogram.max() / cumsum.max()

        # Create masked array to modify each pixel faster
        masked_cumsum = np.ma.masked_equal(cumsum, 0)

        masked_cumsum = (masked_cumsum - masked_cumsum.min()) * 255 / (masked_cumsum.max() - masked_cumsum.min())
        cumsum = np.ma.filled(masked_cumsum, 0).astype('uint8')

        out = cumsum[self.img]
        return out

    def denoise_image(self,weight=10, error_tolerance=1e-3):
        """
            Denoises every channel in an image.
        """
        b = self.denoise_channel(self.img[:, :, 0], weight=weight, error_tolerance=error_tolerance)
        g = self.denoise_channel(self.img[:, :, 1], weight=weight, error_tolerance=error_tolerance)
        r = self.denoise_channel(self.img[:, :, 2], weight=weight, error_tolerance=error_tolerance)
        return cv2.merge((b,g,r))

    def denoise_channel(self, channel, weight=0.1, error_tolerance=1e-3, iterations=200):
        """
            Denoises a channel using ROF algorithm.
        """
        u = np.zeros(channel.shape)
        px = np.zeros(channel.shape)
        py = np.zeros(channel.shape)

        nm = np.sqrt(np.prod(channel.shape[:2]))
        tau = 0.125

        i = 0
        while i < iterations:
            u_old = u

            # Gradient of U
            ux = gradient(u, 1)
            uy = gradient(u, 0)

            # Update variables
            px_new = px + (tau / weight) * ux
            py_new = py + (tau / weight) * uy

            # Compute p's divergence
            norm = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
            px = px_new / norm
            py = py_new / norm

            diverg = divergence(px, 1) + divergence(py, 0)
            u = channel + weight * diverg

            # RMSerr
            rms_error = np.linalg.norm(u - u_old) / nm

            if i == 0:
                err_init = rms_error
                err_prev = rms_error
            else:
                # break if error small enough
                if np.abs(err_prev - rms_error) < error_tolerance * err_init:
                    break
                else:
                    err_prev = rms_error

            i += 1

        return u

    def balance_channel(self, channel, cutoff=5):
        """
            Applies GIMP's white balancing algorithm.
            Cuts off the low cutoff% values the channel,
            and then stretches the image along the new
            range, creating histogram gaps.
        """

        # low value - cutoff% of the array are lower than this value
        # high value - 100-cutoff% of the array are lower than this value
        #

        low = np.percentile(channel, cutoff)
        high = np.percentile(channel, 100 - cutoff)

        # (high - low) is the new range, basically cutoff% are
        # cut off from each end, because too little pixels have
        # them.
        new_channel = ((channel - low) * 255.0 / (high - low))

        # Convert back to uint8
        channel = np.uint8(np.clip(new_channel, 0, 255))
        return channel

    def balance_white(self, cutoff):
        """
            Wrapper for white balance algorithm
            to balance all channels of an image.
        """
        b = self.balance_channel(self.img[:,:,0], cutoff)
        g = self.balance_channel(self.img[:,:,1], cutoff)
        r = self.balance_channel(self.img[:,:,2], cutoff)
        return cv2.merge((b,g,r))







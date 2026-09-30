"""Text-colour isolation on the palette layouts DVDs use."""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from vobsub_to_srt.segment import crop, fill_mask
from vobsub_to_srt.vobsub import Cue

L1, L2 = "The quick brown fox", "jumps over the lazy dog"


def _font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 36)
    except OSError:
        return ImageFont.load_default(36)


def _text(lines):
    im = Image.new("L", (720, 140), 0)
    d = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        d.text((40, 20 + 50 * i), t, font=_font(), fill=255)
    return np.array(im) > 128


def _ring(m, px):
    return ndimage.binary_dilation(m, iterations=px) & ~m


def _scheme(kind):
    """Returns (image, alpha, true text mask) for one palette layout."""
    t = _text([L1, L2])
    img = np.zeros(t.shape, np.uint8)
    alpha = [0, 15, 15, 15]
    if kind == "outline+ring":                 # the common DVD layout
        img[_ring(t, 3)] = 3; img[_ring(t, 1)] = 2; img[t] = 1
    elif kind == "outline only":
        img[_ring(t, 2)] = 3; img[t] = 1
    elif kind == "no outline":
        img[t] = 1; alpha = [0, 15, 0, 0]
    elif kind == "thick ring":                 # 2 px anti-alias ring, no exposed fill edges
        img[_ring(t, 4)] = 3; img[_ring(t, 2)] = 2; img[t] = 1
    elif kind == "ring larger than fill":      # thin font: the ring has more pixels than the fill
        img[_ring(t, 3)] = 3; img[_ring(t, 2)] = 2; img[t] = 1
    elif kind == "box":                        # opaque backdrop behind the text
        img[:] = 1; img[_ring(t, 2)] = 3; img[t] = 2
    elif kind == "box, no outline":
        img[:] = 1; img[t] = 2; alpha = [0, 15, 15, 0]
    elif kind == "translucent box":            # backdrop below the opacity threshold
        img[:] = 1; img[_ring(t, 2)] = 3; img[t] = 2; alpha = [0, 4, 15, 15]
    elif kind == "two colours":                # speaker colours: one per line
        img[_ring(t, 2)] = 3; img[_text([L1, ""])] = 1; img[_text(["", L2])] = 2
    return img, alpha, t


SCHEMES = ["outline+ring", "outline only", "no outline", "thick ring", "ring larger than fill",
           "box", "box, no outline", "translucent box", "two colours"]


@pytest.mark.parametrize("kind", SCHEMES)
def test_isolation_finds_the_text(kind):
    img, alpha, truth = _scheme(kind)
    mask = fill_mask(Cue(0, 0, 1000, img, [(0, 0, 0)] * 4, alpha))
    expect = crop(truth)
    assert mask.shape == expect.shape, kind
    assert np.array_equal(mask, expect), kind


def test_empty_cue():
    assert fill_mask(Cue(0, 0, 1, np.zeros((0, 0), np.uint8), [(0, 0, 0)] * 4, [0, 15, 15, 15])).size == 0
    assert fill_mask(Cue(0, 0, 1, np.zeros((5, 5), np.uint8), [(0, 0, 0)] * 4, [0, 15, 15, 15])).size == 0

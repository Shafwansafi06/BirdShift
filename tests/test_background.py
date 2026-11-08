from PIL import Image
import numpy as np

from birdshift.transforms.background import background_variant, bbox_to_pixels, SquarePad


def test_preserves_bounding_box():
    a = np.random.default_rng(123).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    image = Image.fromarray(a)
    bbox = (20, 10, 45, 35)
    l, t, r, b = bbox_to_pixels(bbox, image.size)
    for kind in ("original", "blur", "replace"):
        changed = np.asarray(background_variant(image, bbox, kind))
        assert np.array_equal(changed[t:b, l:r], a[t:b, l:r])
        if kind != "original":
            assert not np.array_equal(changed[:t, :, :], a[:t, :, :])
    padded = SquarePad()(image)
    assert padded.size == (120, 120)


if __name__ == "__main__":
    test_preserves_bounding_box()
    print("Background transform smoke test passed")

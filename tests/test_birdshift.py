"""BirdShift dataset: box pixels preserved, and val images identical to full-image mode."""
import unittest
import tempfile
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from birdshift.datasets.cub import CUBDataset
from birdshift.transforms.background import background_variant, bbox_to_pixels


class TestBirdShift(unittest.TestCase):
    def test_background_preserves_bbox(self):
        image = Image.new('RGB', (80, 60), (240, 50, 40))
        bbox = (10, 15, 30, 25)
        area = bbox_to_pixels(bbox, image.size)
        for kind in ('blur', 'replace'):
            shifted = background_variant(image, bbox, kind=kind)
            self.assertEqual(list(image.crop(area).getdata()), list(shifted.crop(area).getdata()))

    def test_modes_and_validation_equivalence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'images').mkdir()
            image_lines, class_lines, split_lines, bbox_lines = [], [], [], []
            # 40 samples and two classes, enough to stratify the training set.
            for idx in range(1, 41):
                filename = f'{idx}.jpg'
                Image.new('RGB', (96, 72), (idx * 3 % 255, 60, 90)).save(root/'images'/filename)
                image_lines.append(f'{idx} {filename}')
                class_lines.append(f'{idx} {1 if idx <= 20 else 2}')
                split_lines.append(f'{idx} {1 if idx % 5 != 0 else 0}')
                bbox_lines.append(f'{idx} 10 10 60 40')
            for name, lines in [('images.txt',image_lines), ('image_class_labels.txt',class_lines),
                                ('train_test_split.txt',split_lines), ('bounding_boxes.txt',bbox_lines)]:
                (root/name).write_text('\n'.join(lines))
            full = CUBDataset(root, split='val', mode='full')
            birdshift = CUBDataset(root, split='val', mode='birdshift', suppression_prob=1)
            crop = CUBDataset(root, split='val', mode='crop')
            self.assertEqual(len(full),len(birdshift))
            self.assertTrue(torch.equal(full[0]['image'],birdshift[0]['image']))
            self.assertEqual(tuple(crop[0]['image'].shape),(3,224,224))
            self.assertEqual(tuple(CUBDataset(root,split='train',mode='birdshift',suppression_prob=1)[0]['image'].shape),(3,224,224))


if __name__ == '__main__':
    unittest.main()

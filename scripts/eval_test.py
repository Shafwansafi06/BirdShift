"""Top-1 on CUB official test split for full-image checkpoints."""
import sys, torch
from torch.utils.data import DataLoader
from birdshift.datasets.cub import CUBDataset
from birdshift.models.resnet import BirdResNet

dev = torch.device("cuda")
for run in sys.argv[1:]:
    ck = torch.load(f"checkpoints/{run}/best.pth", map_location=dev, weights_only=False)
    m = BirdResNet(200, pretrained=False).to(dev)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    ds = CUBDataset("data/raw/CUB_200_2011", split="test", mode="full", image_size=ck.get("image_size", 224))
    correct = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in DataLoader(ds, batch_size=64, num_workers=8):
            correct += (m(b["image"].to(dev)).argmax(1).cpu() == b["label"]).sum().item()
    print(f"{run}: epoch {ck['epoch']} val {ck['val_accuracy']:.2%} test {correct/len(ds):.2%} ({correct}/{len(ds)})", flush=True)

from utils.general import strip_optimizer

strip_optimizer("../runs/train/exp-swinT-p1-40-map-384-nc3/weights/last.pt",
                "../weights/exp-swinT-p1-40-map-384-nc3-half.pt")

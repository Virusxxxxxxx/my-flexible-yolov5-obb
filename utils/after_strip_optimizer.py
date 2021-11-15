from utils.general import strip_optimizer

strip_optimizer("../runs/train/exp/weights/last.pt",
                "../weights/exp-swinT-p3-30-map64.8-384-adam-focal-nc3.pt")

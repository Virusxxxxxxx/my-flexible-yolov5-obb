from utils.general import strip_optimizer

strip_optimizer("../weights/exp-swinT-p1-45-map68.5-384-adam-focal.pt",
                "../weights/swinT.pt")

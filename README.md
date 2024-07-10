# Unofficial PyTorch implementation of paper: "Learning to optimise general TSP instances"
The code was written in order to reproduce the model proposed in paper: Sultana et al., Learning to optimise general TSP instances, _Int. J. Mach. Learn. Cybern_., 2022.

Only training code available for now. Please be noted there might be some errors as the repository is not complete and any contributions or issue raisings are welcome. Data for TSP10 is available for now due to size limitations. Data with different size will be available on requests.

---

Please use `conda env create --file environment.yaml` for dependency.

---

Example run (training):
```bash
python train.py fit --config default_config.yaml
```
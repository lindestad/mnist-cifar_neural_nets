# From Perceptrons to Residual Networks  
*A comparative study on MNIST & CIFAR-10*

This repository contains all the code, trained weights and utility scripts used
in the report **“From Perceptrons to Residual Networks: A Comparative Study on
MNIST and CIFAR-10.”**  
It reproduces the full experimental pipeline—from data download to figures and
per-class metric tables—using **PyTorch ≥ 2.2** and **scikit-learn ≥ 1.4**.

---

## 1 Quick start

```bash
# 0) create an environment and install deps
python -m pip install -r requirements.txt

# 1) sanity-check the raw datasets
python visualize_samples.py mnist   # saves images/mnist/samples.png
python visualize_samples.py cifar   # saves images/cifar/samples.png

# 2) train + evaluate any model                                (≈ 6-30 min/GPU)
python train_eval.py --dataset mnist  --model small_mlp
python train_eval.py --dataset mnist  --model simple_cnn
python train_eval.py --dataset cifar  --model large_mlp
python train_eval.py --dataset cifar  --model resnet

# 3) generate saliency maps for the CNNs 
python saliency.py --dataset mnist  --model simple_cnn --n 6
python saliency.py --dataset cifar --model resnet --n 6
```

All outputs (learned weights, loss/accuracy curves, JSON classification reports,
saliency PNGs) are written to:

```
outputs/<dataset>/<model>/
├── best.pt              # model state_dict
├── loss_curve.png       # training & validation loss
├── acc_curve.png        # training & validation accuracy
├── classification.json  # per-class precision/recall/F1
└── saliency/            # heat-maps (only if saliency.py is run)
```

---

## 2 Repository layout

```
.
├── data_utils.py         # OpenML download, preprocessing, augmentation
├── models.py             # Small/large MLP, Simple CNN, ResNet-18
├── train_eval.py         # training loop + early stopping
├── visualize_samples.py  # 5×5 sample grids for sanity-check
├── saliency.py           # gradient-based saliency maps
├── outputs/              # created automatically
├── requirements.txt
└── README.md
```

---

## 3 Re-creating the LaTeX tables

Each training run dumps a `classification.json`.  
A helper in `scripts/make_tables.py` converts these into `tables/*.tex`, which
are `\input{}`-ed by the report:

```bash
python scripts/make_tables.py outputs/mnist/simple_cnn/classification.json \
       --out tables/mnist_simple_cnn.tex
```

---

## 4 Citation

If you use (or adapt) this code for academic work, please cite the accompanying
course report:

```
@misc{lindestad2025perceptrons,
  author    = {Daniel Lindestad},
  title     = {{From Perceptrons to Residual Networks:}
               A Comparative Study on MNIST and CIFAR-10},
  year      = {2025},
  note      = {IKT112 Concepts of Machine Learning, University of Agder}
}
```

---

## 5 License

Source code is released under the MIT License (see `LICENSE` file).  
The MNIST and CIFAR-10 datasets are distributed under their respective
original terms.

# Model Building - Angel Thu Do

---

## Environment Setup

```bash
conda create -n test python=3.10.13
conda activate test   # or: source activate test
cd /your/filepath/././test/
pip install -r requirements.txt
```

## Run Code

```bash
python model.py --data-dir ./Test2 --out-dir ./results
```

## Code Folder Structure
```
test/
├── model.py
├── requirements.txt
├── Test2/
│ ├── BC1_trad_homogenous_2023Nov14_MASKED_new1.csv
│ ├── BC1_trad_homogenous_2023Nov14_MASKED_new2.csv
│ ├── BC1_trad_homogenous_2023Nov14_MASKED_new3.csv
│ ├── BC1_trad_homogenous_2023Nov14_MASKED_new4.csv
│ ├── BC1_trad_homogenous_2023Nov14_MASKED_new5.csv
│ └── BC1_trad_homogenous_2023Nov14_MASKED_new6.csv
├── results/
│ ├── best_model.pkl
│ ├── model_dashboard.png
│ ├── model_metrics.csv
│ ├── performance_by_segment_time.csv
│ ├── performance_by_time.csv
│ ├── plot_01_model_comparison_auc.png
│ ├── ......
│ └── summary.txt
└── README.md
```


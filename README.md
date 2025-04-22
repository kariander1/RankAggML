
# RankAggML: Rank Aggregation with ML-Based Imputation

This project explores the problem of **rank aggregation**—combining multiple ranked lists into a single consensus ranking—under conditions of partial or missing data. It implements and evaluates classical algorithms such as:

- **Naive Aggregation**
- **Fagin’s Algorithm**
- **Threshold Algorithm (TA)**
- **No Random Access (NRA)**

To address inefficiencies in NRA, particularly when the number of desired results \( k \) is large, we introduce a variant called **NRA w/ Imputer**, which uses imputation to estimate missing scores and enable earlier stopping.


## ⚙️ Installation

To get started, ensure you're using **Python 3.8**, and follow these steps:

1. **Create a virtual environment (recommended):**

```bash
python3.8 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

This will install all necessary libraries including `pandas`, `scikit-learn`, `matplotlib`, `numpy`, and more as needed for running aggregation experiments and visualizations.

## 🧪 Running the Code

You can run any experiment using a YAML config file or by explicitly passing arguments on the command line.

#### ✅ Option 1: Run with a Config File

```bash
python main.py --config configs\config_toy_fagin.yaml
```

Example for NRA w/impute algorithm:

```bash
python main.py --config configs\exp_all_algs\config_nra_impute.yaml
```
#### ✅ Option 2: Run with Explicit Arguments

```bash
python main.py \
  --agg_function_name avg \
  --aggregator_name fagin \
  --dataset_name toy \
  --imputer_name basic \
  --p_erase 0.5
  --k 3 \
  --seed 42 \
  --output_path outputs/toy_fagin_p_0_5
```


> 💡 For a full list of supported arguments and defaults, refer to [`config.py`](./config.py), which defines all experiment options using a structured `dataclass`.

---

### 📂 Output Format

After running an experiment, results will be saved inside the folder specified by `output_path`. You will find:

- `results.csv`: Contains the number of **sorted accesses** and **random accesses** for each algorithm and value of \( k \).
- `metrics.csv`: Includes evaluation metrics such as **set accuracy** (correct top-$k$ elements) and **exact match rate** (correct order among top-$k$).



---

## 📊 Data & Visualization

To generate infographics about distribution, you can run:

```bash
python data/visualize.py
```
The dataset is taken from https://www.kaggle.com/datasets/gregorut/videogamesales/data.

> The visualizations are saved in the `outputs/data` directory.


---

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{yehezkel2025mlrankagg,
  title={Rank Aggregation with ML-Based Imputation},
  author={Shai Yehekzel},
  year={2025},
  howpublished={\url{https://github.com/kariander1/RankAggML}},
  note={Accessed: 2025-04-21}
}
```

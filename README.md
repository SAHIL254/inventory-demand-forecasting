# 📦 Inventory Demand Forecasting

Retailers face significant losses due to inaccurate demand estimation, leading to **stockouts** or **overstock situations**. Traditional forecasting approaches often fail to adapt to dynamic market conditions.

Accurate demand forecasting helps retailers:

* Avoid stockouts 📉
* Reduce overstock 📦
* Improve supply chain efficiency 🚚

---

## 📊 Dataset Description

The dataset contains historical sales records with the following features:

| Column | Description                                |
| ------ | ------------------------------------------ |
| Date   | Date of sale                               |
| Store  | Store ID                                   |
| Item   | Product/Item ID                            |
| Sales  | Number of items sold (**target variable**) |

---

## 🧠 Problem Statement

> Predict future sales for each store-item combination using historical data.

This is a **time series forecasting problem** with multiple entities (Store × Item).

---

## 📁 Project Structure

```bash
inventory-demand-forecasting/
│
├── artifacts/                         
│   ├── raw_data.csv
│   ├── processed_inventory_demand.csv   
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── predictions.csv
│
├── notebooks/                        # Jupyter notebooks (experiments)
│   ├── data/
│   │   └── train.csv
│   │
│   ├── 1_EDA_Inventory.ipynb       
│   ├── 2_Model_Training.ipynb        
│
├── src/                              
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── feature_engineering.py   
│   │   ├── model_trainer.py         
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── app/                              
│   ├── app.py                        # Flask
│   ├── templates/
│   └── static/
│
├── config/
│   └── config.yaml                   # paths
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔍 Current Work

* Exploratory Data Analysis (EDA)
* Understanding sales patterns
* Feature engineering for time-based data

---

## 📈 Key Insights (from EDA)

* Sales vary across stores and items
* Strong time-based patterns (trend + seasonality)
* Each Store–Item pair behaves differently
* Weekly and monthly demand cycles exist

---

## ⚙️ Feature Engineering

From `Date`, we extract:

* Year
* Month
* Day
* Day of Week
* Week of Year

Additional strategies:

* Store-level demand patterns
* Item-level behavior
* Interaction features (Store × Item)

---


## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

---



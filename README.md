# inventory-demand-forecasting
Retailers face significant losses due to inaccurate  demand estimates, causing stockouts or  overstock situations. Traditional forecasting  models struggle with dynamic market  conditions .


# Project Structure
Inventory-Demand-Forecasting/
│
├── artifacts/                        # Generated outputs
│   ├── raw_data.csv
│   ├── processed_data.csv 
│   ├── model.pkl
│   └── metrics.json
│
├── notebooks/                        # Experimentation
│   ├── data/
│   │    └── train.csv
│   │
│   ├── 1_EDA_Inventory_Demand.ipynb
│   ├── 2_Feature_Engineering.ipynb
│   └── 3_Model_Training.ipynb
│
├── src/
│   ├── components/
│   │
│   │    ├── data_ingestion.py        # Load & split dataset
│   │    ├── data_transformation.py   # Feature engineering (VERY IMPORTANT)
│   │    ├── model_trainer.py         # Train ML models
│   │    ├── model_evaluation.py      # Evaluate performance
│   │
│   ├── pipeline/
│   │    ├── training_pipeline.py     # End-to-end training
│   │    └── prediction_pipeline.py   # Future demand prediction
│   │
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── app/                             
│   ├── app.py                        # Streamlit / FastAPI app
│   └── templates/
│
├── config/
│   ├── config.yaml                   # Model parameters
│   └── schema.yaml                   # Data schema
│
├── requirements.txt
├── setup.py
└── README.md
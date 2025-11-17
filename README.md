# predictify-commerce-ai
AI-powered predictive analytics system for e-commerce sales forecasting and business insights

# Dependencies
Several packages are required for the development or running the resulting app in general, you can just proceed by running the following command on any terminal, assuming that python3 (within the pip3) installation is already available
```sh
pip install -r requirements.txt
```

# Note
We tried our best to explain and comment all codes in technical english, so we all will likely to see english everywhere except for the user interface app of the dashboard and the final doc report, which will be in french abviously.

The script for regenerating models is also available in this repository.
All models, report and cleaned dataset will be in the **models/** folder 


# Running the notebook
To run the notebook, just type on the terminal
```sh
jupyter notebook
```

Or to make it simple (train the model and run the streamlit app), we just do
```sh
python train.py
streamlit run app.py
```
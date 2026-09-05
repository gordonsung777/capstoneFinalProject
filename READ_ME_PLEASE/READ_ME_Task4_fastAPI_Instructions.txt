'''Task 4b) Create backend endpoints using FastAPI for model inference and log insights'''
'''
Again, run this in Visual Code:

find the python code:   digital_bank_fast_api.py

make sure to have:

pip install Uvicorn  --> With Uvicorn, digital_bank_fastapi:app will load the app object from the digital_bank_fastapi.py model

pip install "fastapi[standard]" pandas scikit-learn --> install FastAPI

run --> uvicorn digital_bank_fastapi:app --reload


then go to http://127.0.0.1:8000/docs 

FastAPI will auto generate intereactive API documentation:

click POST /predict-criticality and click Try it Out--> enter the KPI values, and click Execute



so main purpose is that FastAPI becomes the backend/model server layer, and Streamlit helps displays the result in more elegant GUI format for 
viewers/users to easily see and read
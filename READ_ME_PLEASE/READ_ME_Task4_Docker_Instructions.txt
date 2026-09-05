

'''4c) Containerize components with Docker and orchestrate using Docker Compose'''
please find: 
compose.yaml
Dockerfile.api
Dockerfile.streamlit

Open the software app Docker Desktop you installed, so its running on your computer:
Then, open powershell, and type: 

docker info 

to make sure it is working on your computer

then run:
docker compose up --build

once it is completed building the docker file, go onto http://localhost:8501/ for Streamlit GUI and http://localhost:8000/docs for API GUI
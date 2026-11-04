#pull the python image from dockerhub
#use a lightweight python image
FROM python:3.14-slim

#set the working directory inside the container
WORKDIR /app

#copy dependency list before copying the application code
COPY requirements.txt .

#install the dependencies 
RUN pip install -r requirements.txt

#copy application code and saved model inside /app
COPY . .

#document the port used by the fastapi application
EXPOSE 8000

#start the api server when container is ready
# 0.0.0.0 - allows connections through the container's network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.7.8-slim as frontend
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y make automake gcc g++ subversion python3-dev
COPY ./requirements.txt /apps/requirements.txt
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install --requirement /apps/requirements.txt
COPY src/. /apps
WORKDIR /apps
RUN python3 --version
RUN ls /apps

CMD ["python3.7", "app.py"]

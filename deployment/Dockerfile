FROM python:3.11-slim

RUN pip install -U pip --upgrade pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["flask-app.py", "best_model_scaler_KNN.bin", "best_model_KNN.bin", "./"]
# COPY ["flask-app.py", "./"]

EXPOSE 9696

ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind=0.0.0.0:9696", "flask-app:app"]

import requests

url = 'http://127.0.0.1:9696/predict'

data = {
    "age": 54.366337,
    "sex": 0.683168,
    "cp": 0.966997,
    "trtbps": 131.623762,
    "chol": 246.264026,
    "fbs": 0.148515,
    "restecg": 0.528053,
    "thalachh": 149.646865,
    "exng": 0.326733,
    "oldpeak": 1.039604,
    "slp": 1.39934,
    "caa": 0.729373,
    "thall": 2.313531
}

response = requests.post(url, json=data).json()
print(response)
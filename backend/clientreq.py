import requests

url = "http://127.0.0.1:5000/postpred"

# response = requests.get(url)
files = {"file":open("images/Amitabh Bachchan_0.jpg","rb")}
resp = requests.post(url,files=files)
print(resp.json())
print(resp.status_code)
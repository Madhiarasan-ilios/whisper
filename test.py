import requests

url = "http://10.0.27.169:5000/transcribe"  # Replace with your actual public IP

files = {
    'audio': open('sample.wav', 'rb')  # Replace with your audio file path
}

data = {
    'token': 'your_huggingface_token_here'  # Replace with your HF token
}

response = requests.post(url, files=files, data=data)

print("Status Code:", response.status_code)
print("Response Text:\n", response.text)

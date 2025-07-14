import requests
import base64
 
url = "http://localhost:8080/score_local?get_final_text=1&render_final_text=1"
file_path = "/home/w-4090/cutted_score_audio_separated/446892/344523004.wav"
 
response = requests.post(url, data={"path": file_path})

resp = response.json()
if "speech" in resp and "data" in resp["speech"]:
    data = resp["speech"]["data"]
    file_to_save = open("test_submit.mp3", "wb")
    file_to_save.write(base64.b64decode(data))
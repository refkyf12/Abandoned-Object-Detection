import socket
import json
import requests

config_file = "config.json"

with open(config_file, "r") as jsonfile:
    config = json.load(jsonfile)

bot_token = config["bot_token"]
url_photo = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
url_message = f"https://api.telegram.org/bot{bot_token}/sendMessage"

def notification_tele(message, img_name, img_path):
    for chat_id in config['tele_receriver']:
        try:
            
            file_contents = None
            with open(img_path, "rb") as file:
                file_contents = file.read()
            
            response = requests.post(
                url_photo, 
                data={
                    "chat_id": chat_id
                },
                files={
                    "photo": (img_name, file_contents),
                },
            )
            
            if response.status_code != 200:
                print("Error sending image:", response.text)
            else:
                file_id = response.json()["result"]["photo"][-1]["file_id"]
                
                # Send the text message with the image using the Telegram API
                response = requests.post(
                    url_message,
                    data={
                        "chat_id": chat_id,
                        "text": message,
                        "photo": file_id,
                    },
                )
                
                # Check if the message was sent successfully
                if response.status_code != 200:
                    print('---------------------------------')
                    print("Error sending message:", response.text)
                    print('---------------------------------')
                else:
                    print('---------------------------------')
                    print("Message sent successfully!")
                    print('---------------------------------')

        except socket.error as e:
            print('---------------------------------')
            print('Notification Telegram Error')
            print('---------------------------------')
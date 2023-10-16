from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path

import smtplib, socket
import json

config_file = "config.json"

with open(config_file, "r") as jsonfile:
    config = json.load(jsonfile)

def notification_email(body, img_path):

    print('Sending Notification Email ....')
    
    for email in config['email_receriver']:
        try:
            
            #The mail addresses and password
            sender_address = 'aylingaura@gmail.com'
            sender_pass = 'gslnwdcuyrnwfspa'
            receiver_address = email

            message = MIMEMultipart()
            message["from"] = "Rastek Inovasi Digital"
            message["to"] = receiver_address
            message["subject"] = "Alert Notification"
            
            html_body = """\
                        <html>
                        <head></head>
                        <body>
                            """+ str(body) +"""
                        </body>
                        </html>
                        """
            
            message.attach(MIMEText(html_body, 'html'))
            
            if img_path is not None :
                message.attach(MIMEImage(Path(img_path).read_bytes()))
            
            with smtplib.SMTP(host="smtp.gmail.com", port=587) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(sender_address, sender_pass)
                smtp.send_message(message)
                
            print('Notification Email Send')
        except socket.error as e:
            print('Notification Email Error')

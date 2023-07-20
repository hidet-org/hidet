import smtplib
from email.mime.text import MIMEText
import getpass


class EmailSender:
    def __init__(self) -> None:
        self.recipients = input('Enter comma separated recipient email addresses:\n')
        self.recipients = self.recipients.replace(' ', '').split(',')
        self.sender = input('Enter your Gmail address:\n')
        self.password = getpass.getpass(prompt='Enter your 16-digit Google app password: ')

    def send_email(self, body):
        subject = 'Hidet Performance Regression'
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = ', '.join(self.recipients)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(self.sender, self.password)
            smtp_server.sendmail(self.sender, self.recipients, msg.as_string())
        print("Results sent to", msg['To'])


# email_utils.py
import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv

def send_email(receiver: str, subject: str, body: str) -> str:
    """Send an Email to a given recipient with a subject and message"""
    load_dotenv()

    sender_email = os.getenv("SENDER_EMAIL")
    app_password = os.getenv("APP_PASSWORD")

    if not sender_email or not app_password:
        return "❌ Error: Sender email or app password not found in environment variables."

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return "✅ Email sent successfully! write the inform to confirm"
    except Exception as e:
        return f"❌ Error sending email: {e}"

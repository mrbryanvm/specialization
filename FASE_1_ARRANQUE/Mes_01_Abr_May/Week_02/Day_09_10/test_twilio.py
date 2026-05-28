import os

from dotenv import load_dotenv
from twilio.rest import Client

# 1. Load environment variables from .env file
load_dotenv()

# 2. Load credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_NUMBER")
user_number = os.getenv("USER_NUMBER")

# 3. Validate — if ANY variable is missing, crash early with a clear message
if not account_sid or not auth_token or not twilio_number or not user_number:
    print("❌ ERROR: Missing credentials in your .env file.")
    exit(1)

# At this point, Python KNOWS all variables are strings (not None)
# because we already exited if any were None.
assert isinstance(account_sid, str)
assert isinstance(auth_token, str)
assert isinstance(twilio_number, str)
assert isinstance(user_number, str)

# 4. Initialize the Twilio client
client = Client(account_sid, auth_token)

try:
    print(f"Sending test message from {twilio_number} to {user_number}...")

    # 5. Send the WhatsApp message
    message = client.messages.create(
        body="Hello Bryan! This message was sent automatically from my Python script. 🐍",
        from_=twilio_number,
        to=user_number,
    )

    # 6. Confirmation
    print("✅ Message sent successfully!")
    print(f"Message SID: {message.sid}")

except Exception as e:
    print(f"❌ Error sending message: {e}")

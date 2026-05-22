import os
from dotenv import load_dotenv
from twilio.rest import Client

# 1. Cargar las variables de entorno desde el archivo .env
load_dotenv()

# 2. Obtener las credenciales
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_NUMBER")
user_number = os.getenv("USER_NUMBER")

# Validar que las credenciales no estén vacías
if not account_sid or "TU_AUTH_TOKEN" in auth_token or not auth_token:
    print("❌ ERROR: Por favor configura correctamente tu archivo .env con tus credenciales de Twilio.")
    exit(1)

# 3. Inicializar el cliente de Twilio
client = Client(account_sid, auth_token)

try:
    print(f"Enviando mensaje de prueba desde {twilio_number} hacia {user_number}...")
    
    # 4. Enviar el mensaje de WhatsApp
    message = client.messages.create(
        body="¡Hola Bryan! Este es un mensaje enviado automáticamente desde mi script de Python. 🚀🐍",
        from_=twilio_number,
        to=user_number
    )
    
    # 5. Confirmación
    print("✅ ¡Mensaje enviado con éxito!")
    print(f"SID del mensaje: {message.sid}")

except Exception as e:
    print(f"❌ Ocurrió un error al enviar el mensaje: {e}")

import requests

# 1. Hacemos una petición GET a esta URL
url = "https://jsonplaceholder.typicode.com/users/1"
response = requests.get(url)

# 2. Verificamos el código de estado (200 significa OK)
print(f"Status Code: {response.status_code}")

# 3. Convertimos la respuesta directamente a un diccionario de Python usando .json()
data = response.json()

# 4. Imprimimos algunos datos del diccionario
print(f"User name: {data['name']}")
print(f"User email: {data['email']}")

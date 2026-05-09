contador = 5
print("Iniciando cuenta regresiva")
while contador > 0:
    print(f"T-{contador}")
    contador = contador -1
print("DESPEGUE")

frutas = ["Manzana", "Frutilla", "Naranja", "Pera"]
print("\nLista de compras:")
for fruta in frutas:
    print(f"- {fruta}")


# examples to practice
# Counter app that count from 50 to 1 with interval of 5
#  show in red the numbers multiples of 10
#  show in blue the numbers multiples of 5
counter = 50
print("\nStarting countdown...")
while(counter>=1):
    print(counter)
    counter=counter-5
print("Ended")

#Fruits
fruits = ["apple", "Banana", "Grapes", "Orange"]
print("\nPrinting fruits...")
for fruit in fruits:
    print(f"-{fruit}")

#Cars
brands = {"Toyota", "Suzuki", "Fiat", "Ford", "BMW", "Mercedez Benz", "Audi"}
print("\nCar brands:")
for brand in brands:
    print(f"-{brand}")

#Increment age by 5 years until 30 years old
print("\nAge increment:")
age = 0
while age<=30:
    print(age)
    age = age + 5
print("You are 30 years old!")


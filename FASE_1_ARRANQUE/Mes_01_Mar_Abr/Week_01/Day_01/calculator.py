# Calculadora Interactiva - Día 1
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b

def calculator():
    print("INTERACTIVE CALCULATOR")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    option = input("Choose an option (1-4); ")

    num1 = float(input("Enter first number: "))
    num2 = float(input("Enter second number: "))

    if option == "1":
        result = add(num1, num2)
        print(f"Result: {result}")
    elif option == "2":
        result = subtract(num1, num2)
        print(f"Result: {result}")
    elif option == "3":
        result = multiply(num1, num2)
        print(f"Result: {result}")
    elif option == "4":
        result = divide(num1, num2)
        print(f"Result: {result}")
    else:
        print("invalid option")

if __name__ == "__main__":
    calculator()
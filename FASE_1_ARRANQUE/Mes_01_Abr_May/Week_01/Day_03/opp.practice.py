class Product:
    def __init__(self, name, price, stock):
        self.name = name    
        self.price = price
        self.stock = stock

    def sell(self, quantity):
        if quantity > 0 and self.stock >= quantity:
            self.stock -= quantity
            return True
        print("Not enough stock to complete the purchase")
        return False

    def display_info(self):
        return f"Product: {self.name} - Price: ${self.price} - Stock: {self.stock}"

laptop = Product("Laptop", 1200, 5)
laptop.sell(2)
print(laptop.display_info())
print(laptop.stock)
class Product:
    """
    Represents a product in an inventory system.
    """
    def __init__(self, name, price, stock):
        # In Python, we don't need to declare variables outside __init__
        self.name = name
        self.price = price
        self.stock = stock

    def sell(self, quantity):
        """
        Reduces stock if enough units are available.
        """
        if quantity <= self.stock:
            self.stock -= quantity
            print(f"Success: Sold {quantity} units of {self.name}.")
        else:
            print(f"Error: Not enough stock for {self.name}.")

    def display_info(self):
        """
        Prints current product details.
        """
        print(f"Product: {self.name} | Price: ${self.price} | Stock: {self.stock}")


class Product2:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
    
    def sell(self, quantity):
        if quantity > 0 and self.stock >= quantity:
            self.stock -= quantity
            print(f"Success: Sold {quantity} units of {self.name}")
        else:
            print(f"Error: Invalid quantity or not enough stock for {self.name}")
    
    def display_info(self):
        print(f"Product: {self.name} |  Price: {self.price}  |  Stock: {self.stock}")


# Testing the class
print(f"==========================TESTING PRODUCT CLASS==========================")
if __name__ == "__main__":
    # Instantiate the object (No 'new' keyword needed)
    laptop = Product("MacBook Pro", 1500, 10)
    
    # Display initial state
    laptop.display_info()
    
    # Perform a sale
    laptop.sell(3)
    
    # Verify stock reduction
    laptop.display_info()
    
    # Test error case
    laptop.sell(15) 

    # Testing the class Product2
    print(f"==========================TESTING PRODUCT 2 CLASS==========================")
    laptop2 = Product2("Dell Inspiron", 800, 15)
    laptop2.display_info()
    laptop2.sell(5)
    laptop2.display_info()
    laptop2.sell(100)
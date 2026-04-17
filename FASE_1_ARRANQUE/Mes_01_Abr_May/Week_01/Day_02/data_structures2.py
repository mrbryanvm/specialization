#labels = ["sale", "food", "drink", "dessert", "toy", "home", "luxury", "refrigerated", "expiring"]
product = {
    "name": "Laptop",
    "price": 5000,
    "labels": ["luxury", "electronics"]
}
print(f"Selected product: {product['name']}")

product['labels'].append("sale")
print("\n---Product details---")
print(product)
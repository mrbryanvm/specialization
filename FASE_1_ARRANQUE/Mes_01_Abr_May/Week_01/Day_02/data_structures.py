#labels = ["sale", "food", "drink", "dessert", "toy", "home", "luxury", "refrigerated", "expiring"]
product = {
    "name": "Milk",
    "price": 10.90,
    "labels": ["expiring", "drink", "refrigerated"]
}

print(f"Selected product: {product['name']}")

product['labels'].append("lactose_free")

print("\n---Product details---")
print(product)
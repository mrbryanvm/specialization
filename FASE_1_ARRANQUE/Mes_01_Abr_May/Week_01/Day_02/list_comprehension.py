# lista inicial con datos sucios
print(f"===PRIMERA LISTA===")
prices = [15.50, 20.0, -8.0, -74, 27.7, -99.99, 100.0]
# Filtrar valores positivos y sumarles 1
valid_prices = [p + 1.0 for p in prices if p > 0]
# Resultado
print(f"Original prices: {prices}")
print(f"Valid prices: {valid_prices}")

# EJEMPLO 2
print(f"\n\n===SEGUNDA LISTA===")
prices2 = [54,87,98,55,-2,-5,-99,1]
valid_prices2 = [p + 2 for p in prices2 if p > 0]
print(f"Original prices: {prices2}")
print(f"Valid prices: {valid_prices2}")
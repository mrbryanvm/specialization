# Paso 1: Inicializar el acumulador (Como un double total = 0.0 en Java)
total_revenue = 0.0

# Paso 2: Abrir el archivo de forma segura
# 'with' es el "try-with-resources" de Python: cierra el archivo solo
with open('sales.csv', 'r') as file:
    
    # Paso 3: Saltarse la cabecera
    # next(file) consume la primera línea (Date,Product,Price,Quantity) 
    # para que no dé error al intentar convertir texto a número
    next(file)
    
    # Paso 4: Iterar sobre las líneas restantes
    for line in file:
        # .strip() quita el salto de línea (\n)
        # .split(',') divide la cadena en una lista de strings
        parts = line.strip().split(',')
        
        # Paso 5: Extraer y convertir (Casting)
        # En Python, el índice empieza en 0. 
        # Price es índice 2, Quantity es índice 3.
        price = float(parts[2])
        quantity = int(parts[3])
        
        # Paso 6: Calcular y sumar
        total_revenue += (price * quantity)

# Paso 7: Mostrar el resultado con formato
print(f"--- REPORTE DE VENTAS ---")
print(f"Total Revenue del día: ${total_revenue:,.2f}")
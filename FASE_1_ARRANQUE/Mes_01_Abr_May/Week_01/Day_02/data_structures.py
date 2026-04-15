#etiquetas = ["oferta", "comida", "bebida", "postre", "juguete", "hogar", "suntuoso", "refrigerado", "con_vencimiento"]
producto = {
    "nombre": "Leche",
    "precio": 10.90,
    "etiquetas": ["con_vencimiento", "bebida", "refrigerado"]
}

print(f"Producto seleccionado: {producto['nombre']}")

producto['etiquetas'].append("deslactosado")

print("\n---Detalle del producto---")
print(producto)
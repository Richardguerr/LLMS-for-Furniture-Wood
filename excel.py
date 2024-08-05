import pandas as pd
import json

# Lee el archivo Excel
df = pd.read_excel('../dataset/RK_D_MADERAS.xlsx')

# Inicializa una lista para almacenar los objetos JSON
json_data = []

# Itera sobre cada fila del DataFrame
for index, row in df.iterrows():
    # Crea el objeto JSON para cada fila
    data = {
        "prompt": f"Generate a quotation for {row['quantity']} {row['furniture']} made of {row['wood']} with the color {row['paint']}, measuring {row['dimensions']}, and {row['details']} as additional details.",
        "completion": f"The {row['furniture']} costs {row['price']} cop each and is made with high-quality {row['wood']} wood and the color {row['paint']} with {row['details']} as additional details. The total cost for {row['quantity']} {row['furniture']} is ${row['quantity'] * row['price']} cop. The cost breakdown is as follows: ${row['wood_price']} for the wood, ${row['paint_price']} for the paint, ${row['labour']} for the labor, and additional costs of ${row['additional costs']}. For more information, contact us at 315-320-9459 or visit our website at RKMaderas.com.co."
    }
    
    # Agrega el objeto JSON a la lista
    json_data.append(data)

# Guarda los datos en un archivo JSON
with open('../data_training/training_data.json', 'w') as f:
    json.dump(json_data, f, indent=4)

print("Archivo JSON generado correctamente.")

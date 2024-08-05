import pandas as pd
import json

# Cargar el archivo Excel
df = pd.read_excel('../dataset/furniture_amazon.xlsx')

# Inspeccionar las primeras filas para verificar el formato
print(df.head())

# Procesar la columna 'price_cop'
df['price_cop'] = df['price_cop'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Función para procesar la columna 'categories'
def process_categories(x):
    try:
        # Reemplazar comillas simples por dobles para que sea un JSON válido
        return json.loads(x.replace("'", '"'))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for categories: {x}")
        return []

# Aplicar la función a la columna 'categories'
df['categories'] = df['categories'].apply(process_categories)

# Procesar la columna 'package_dimensions'
dimensions = df['package_dimensions'].str.extract(r'(\d+\.\d+)"D x (\d+\.\d+)"W x (\d+\.\d+)"H')
df[['depth', 'width', 'height']] = dimensions.apply(pd.Series)
df['dimensions'] = df[['depth', 'width', 'height']].apply(lambda x: 'x'.join(x.dropna().astype(str)), axis=1)
df = df.drop(columns=['package_dimensions', 'depth', 'width', 'height'])

# Función para procesar la columna 'about_item'
def process_about_item(x):
    try:
        # Reemplazar comillas simples por dobles para que sea un JSON válido
        return json.loads(x.replace("'", '"'))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for about_item: {x}")
        return []

# Aplicar la función a la columna 'about_item'
df['about_item'] = df['about_item'].apply(process_about_item)

# Crear el formato JSON para entrenamiento
training_data = []

for _, row in df.iterrows():
   data = {
    
    "prompt": f"Generate a quotation for a product in the categories {', '.join(row['categories'])}, made of {row['material']} with the color {row['color']}, measuring {row['dimensions']}, and in the {row['style']} style. Additional details: {' '.join(row['about_item'])}.",
    "completion": f"The product costs {row['price_cop']} COP and is made with high-quality {row['material']} and the color {row['color']}. It falls under the categories {', '.join(row['categories'])}. The total cost is {row['price_cop']} COP. Additional details: {' '.join(row['about_item'])}. For more information, contact us at 315-320-9459 or visit our website at RKMaderas.com.co."

}

   training_data.append(data)

# Guardar los datos en un archivo JSON
with open('../data_training/training_data2.json', 'w') as f:
    json.dump(training_data, f, indent=4)

print("Datos de entrenamiento guardados en training_data2.json")

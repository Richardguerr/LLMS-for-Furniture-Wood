from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json

# Función para cargar y combinar datos de múltiples archivos JSON
def load_and_combine_data(files):
    combined_data = []
    for file in files:
        with open(file) as f:
            data = json.load(f)
            combined_data.extend(data)
    return combined_data

# Cargar los datos de entrenamiento desde múltiples archivos JSON
training_files = ['./data_training/training_data.json']
training_data = load_and_combine_data(training_files)


# Preparar los datos de entrenamiento en un formato adecuado para Hugging Face
def preprocess_data(data):
    texts = [f"user: {item['prompt']} bot: {item['completion']}" for item in data]
    return texts

texts = preprocess_data(training_data)
dataset = Dataset.from_dict({"text": texts})

# Tokenizar los datos
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Configurar el token de padding

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)

# Tokenizar el dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Ajustar el modelo para la tarea de lenguaje
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Cargar el modelo GPT-2
model = GPT2LMHeadModel.from_pretrained(model_name)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el tokenizador entrenados
model.save_pretrained("./fine_tuned_model_gpt2")
tokenizer.save_pretrained("./fine_tuned_model_gpt2")

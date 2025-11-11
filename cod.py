import pandas as pd
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from transformers import DataCollatorWithPadding

data = pd.read_csv("escom_chatbot_dataset.csv", encoding='latin1')
print(data.head())
print("Total de filas:", len(data))

model_name = "datificate/gpt2-small-spanish"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def format_row(row):
    return f"Usuario: {row['pregunta']}\nAsistente: {row['respuesta']}"

data["text"] = data.apply(format_row, axis=1)

#print(data["text"].iloc[0])

# Separar datos
train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)

# Crear datasets de Hugging Face con columnas pregunta y respuesta
train_dataset = Dataset.from_pandas(train_df[["pregunta", "respuesta"]])
test_dataset = Dataset.from_pandas(test_df[["pregunta", "respuesta"]])

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print("Columnas reales del dataset:", dataset["train"].column_names)

def tokenize_function(example):
    inputs = tokenizer(
        example["pregunta"], 
        truncation=True, 
        padding="max_length",
        max_length=64
    )
    labels = tokenizer(
        example["respuesta"], 
        truncation=True, 
        padding="max_length",
        max_length=64
    )
    inputs["labels"] = labels["input_ids"]
    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./gpt2-escom",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    push_to_hub=False,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset= tokenized_datasets["train"],
    eval_dataset= tokenized_datasets["test"],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./modelo_escom_chatbot")
tokenizer.save_pretrained("./modelo_escom_chatbot")

chatbot = pipeline("text-generation", model="./modelo_escom_chatbot", tokenizer=tokenizer)

def responder(pregunta):
    prompt = f"Usuario: {pregunta}\nAsistente:"
    respuesta = chatbot(prompt, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return respuesta[0]["generated_text"].split("Asistente:")[-1].strip()

print(responder("¿Dónde puedo consultar mis calificaciones?"))
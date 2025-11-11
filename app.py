import os
from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.data_loader import DataLoader
from chatbot.chatbot_core import ChatbotESCOM


# Cargar archivos de datos
DATA_FILES = {
"becas": "data/becas.csv",
"tramites": "data/tramites.csv",
"horarios": "data/horarios.csv",
"servicios": "data/servicios.csv",
"general": "data/general.csv"
}


# Inicializar DataLoader y Chatbot
loader = DataLoader()
loader.load_csvs(DATA_FILES)
chatbot = ChatbotESCOM(loader)


app = FastAPI(title="Chatbot ESCOM Inteligente", version="1.0")


class Pregunta(BaseModel):
    pregunta: str
    user_id: str = "default_user"


@app.post("/chat")
def responder(data: Pregunta):
    respuesta = chatbot.responder(data.pregunta, user_id=data.user_id)
    return {"respuesta": respuesta}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
from sentence_transformers import SentenceTransformer
from chatbot.nlp_engine import NLPEngine
from chatbot.context_manager import ContextManager

class ChatbotESCOM:
    def __init__(self, data_loader):
        self.model = data_loader.model
        self.data = data_loader.get_all_data()
        self.engine = NLPEngine(self.data)
        self.context = ContextManager()

    def responder(self, pregunta: str, user_id: str = "default_user") -> str:
        pregunta_emb = self.model.encode(pregunta, convert_to_tensor=True)


        # 1. Buscar en todo el dataset
        respuesta, tema, score = self.engine.buscar_respuesta(pregunta_emb, umbral=0.55)


    # 2. Si no encuentra, intenta con contexto previo (umbral más bajo)
        if respuesta is None:
            tema_prev = self.context.obtener_tema(user_id)
            if tema_prev:
                subset = self.data[self.data["tema"] == tema_prev].reset_index(drop=True)
                engine_ctx = NLPEngine(subset)
                respuesta, tema, score = engine_ctx.buscar_respuesta(pregunta_emb, umbral=0.4)


        # 3. Si encontró una respuesta válida, actualiza contexto y devuelve
        if respuesta:
            if tema:
                self.context.actualizar_tema(user_id, tema)
            return respuesta


        # 4. Si no encontró suficiente similitud, devolver mensaje por defecto
        return "Lo siento, solo puedo responder preguntas relacionadas con la ESCOM."
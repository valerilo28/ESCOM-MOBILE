class ContextManager:
    def __init__(self):
        self.contextos = {}

    def obtener_tema(self, user_id):
        return self.contextos.get(user_id, {}).get("tema")

    def actualizar_tema(self, user_id, tema):
        self.contextos[user_id] = {"tema": tema}
    
    def limpiar_tema(self, user_id):
        if user_id in self.contextos:
            del self.contextos[user_id]

from sentence_transformers import util

class NLPEngine:
    def __init__(self, data):
        self.data = data

    def buscar_respuesta(self, pregunta_emb, umbral=0.5):
        # calcula similaridades y devuelve la mejor respuesta y tema
        similitudes = [util.cos_sim(pregunta_emb, emb).item() for emb in self.data["embedding"]]
        if not similitudes:
            return None, None, 0.0
        idx = max(range(len(similitudes)), key=lambda i: similitudes[i])
        mejor_sim = similitudes[idx]
        if mejor_sim < umbral:
            return None, None, mejor_sim
        fila = self.data.iloc[idx]
        return fila["respuesta"], fila["tema"], mejor_sim

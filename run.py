"""
=============================================================
  CHATBOT ADMISIÓN Y NIVELACIÓN UG — EJECUTAR ESTE ARCHIVO
  Uso: python run.py
=============================================================
"""

import subprocess
import sys
import os

# ── 1. INSTALAR DEPENDENCIAS ────────────────────────────────────
print("\n📦 Instalando dependencias...")
paquetes = ["nltk", "scikit-learn", "numpy", "gradio"]
for pkg in paquetes:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        check=False
    )
print("✓ Dependencias listas\n")

# ── 2. IMPORTS ──────────────────────────────────────────────────
import json
import re
import random
import unicodedata
import numpy as np
import nltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Descargar recursos NLTK
for recurso in ["punkt", "stopwords", "punkt_tab"]:
    nltk.download(recurso, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

import gradio as gr

# ── 3. DATOS: INTENCIONES ───────────────────────────────────────
INTENCIONES = [
    {
        "nombre": "preguntar_requisitos_admision",
        "utterances": [
            "cuales son los requisitos para la admision",
            "que necesito para inscribirme en la UG",
            "que documentos piden para la admision",
            "que papeles necesito para ingresar a la universidad",
            "requisitos de ingreso a la universidad de guayaquil",
            "que se necesita para postular a la UG",
            "documentos necesarios para la inscripcion",
        ],
        "respuestas": [
            "Para la admisión a la UG necesitas: 1) Cédula de identidad vigente, "
            "2) Título de bachiller o acta de grado, 3) Foto tamaño carnet, "
            "4) Puntaje del examen SENESCYT, 5) Cuenta creada en admision.ug.edu.ec."
        ]
    },
    {
        "nombre": "consultar_fechas_registro",
        "utterances": [
            "cuando es el registro para la admision",
            "cuales son las fechas de inscripcion",
            "cuando puedo registrarme en la ug",
            "hasta cuando es la inscripcion",
            "fecha limite de registro",
            "cronograma de inscripcion",
            "en que fecha me registro segun mi cedula",
        ],
        "respuestas": [
            "El cronograma de registro está segmentado por el último dígito de tu cédula. "
            "Consulta las fechas exactas en https://admision.ug.edu.ec/admision/ "
            "y sigue el Facebook oficial de Admisión UG para alertas."
        ]
    },
    {
        "nombre": "crear_cuenta_portal",
        "utterances": [
            "como creo mi cuenta en el portal de admision",
            "como me registro en la pagina de la ug",
            "no puedo crear mi cuenta en admision",
            "pasos para crear cuenta en admision ug",
            "como me registro como aspirante",
            "como accedo al portal de admision",
            "como hago mi usuario en la ug",
        ],
        "respuestas": [
            "Para crear tu cuenta: 1) Ve a admision.ug.edu.ec, 2) Clic en 'Crear cuenta', "
            "3) Ingresa tu cédula, 4) Completa tus datos, 5) Verifica tu correo electrónico."
        ]
    },
    {
        "nombre": "consultar_fechas_nivelacion",
        "utterances": [
            "cuando empiezan las clases de nivelacion",
            "cuales son las fechas del curso de nivelacion",
            "cuando es el curso nivelatorio",
            "cuando comienza nivelacion en la ug",
            "fechas del curso de nivelacion universitaria",
            "cuando inicia el nivelatorio",
            "cronograma de nivelacion ug",
        ],
        "respuestas": [
            "Las fechas del curso de nivelación se publican en https://admision.ug.edu.ec/nivelacion/. "
            "El curso inicia después del proceso de admisión y matrícula."
        ]
    },
    {
        "nombre": "aprobar_nivelacion",
        "utterances": [
            "como apruebo el curso de nivelacion",
            "cuantas materias debo aprobar en nivelacion",
            "requisitos para aprobar nivelacion",
            "como funciona la aprobacion en nivelacion",
            "cuantas asignaturas hay en nivelacion",
            "que necesito para terminar nivelacion",
            "criterios de aprobacion del nivelatorio",
        ],
        "respuestas": [
            "Para aprobar nivelación debes: nota mínima aprobatoria en todas las asignaturas, "
            "asistencia mínima del 70-75%, y completar todas las evaluaciones. "
            "Consulta los detalles en https://admision.ug.edu.ec/nivelacion/."
        ]
    },
    {
        "nombre": "consultar_cupo_aceptado",
        "utterances": [
            "como saber si fui aceptado en la ug",
            "donde veo si me dieron cupo",
            "como verifico mi cupo en la universidad",
            "como saber si me aceptaron en la carrera",
            "donde aparece el resultado de la admision",
            "como me entero si pase la admision",
        ],
        "respuestas": [
            "Para verificar tu cupo: 1) Ingresa a admision.ug.edu.ec con tu cuenta, "
            "2) Revisa 'Resultados' o 'Mi estado'. También puedes verificar en www.senescyt.gob.ec."
        ]
    },
    {
        "nombre": "proceso_matricula",
        "utterances": [
            "como me matriculo en la ug",
            "cuando es la matricula",
            "pasos para hacer la matricula",
            "como hago la matricula en la universidad",
            "fechas de matriculacion en la ug",
            "proceso de matricula universitaria",
            "cuando puedo hacer mi matricula",
        ],
        "respuestas": [
            "La matrícula se realiza: 1) Después de aceptar el cupo, 2) En las fechas del "
            "cronograma oficial, 3) A través del portal admision.ug.edu.ec. "
            "Sigue el Facebook de Admisión UG para no perder las fechas."
        ]
    },
    {
        "nombre": "carreras_disponibles",
        "utterances": [
            "que carreras tiene la universidad de guayaquil",
            "cuales son las carreras de la ug",
            "en que puedo estudiar en la ug",
            "oferta academica de la universidad de guayaquil",
            "que facultades tiene la ug",
            "que puedo estudiar en la universidad de guayaquil",
        ],
        "respuestas": [
            "La UG ofrece carreras en: Ciencias Matemáticas y Físicas, Medicina, Derecho, "
            "Ingeniería, Ciencias Económicas, Filosofía, entre otras. "
            "Ve el catálogo completo en https://admision.ug.edu.ec/admision/."
        ]
    },
    {
        "nombre": "puntaje_examen_senescyt",
        "utterances": [
            "cuanto puntaje necesito para entrar a la ug",
            "cual es el puntaje minimo del examen",
            "que nota necesito en el senescyt para la ug",
            "puntaje requerido para admision en la ug",
            "cuanto debo sacar en el examen de admision",
            "nota minima del ser bachiller",
            "cuanto necesito en el examen nacional",
        ],
        "respuestas": [
            "El puntaje mínimo del examen SENESCYT varía por carrera (generalmente 600-800 puntos). "
            "Consulta los puntajes exactos del período vigente en www.senescyt.gob.ec."
        ]
    },
    {
        "nombre": "calificaciones_nivelacion",
        "utterances": [
            "como ver mis notas en nivelacion",
            "donde consulto mis calificaciones del nivelatorio",
            "como saber mis notas del curso de nivelacion",
            "como accedo a mis notas universitarias",
            "sistema de calificaciones en nivelacion",
            "donde veo mi rendimiento academico",
        ],
        "respuestas": [
            "Para ver tus calificaciones: 1) Ingresa a admision.ug.edu.ec/nivelacion/, "
            "2) Accede con tu cuenta institucional, 3) Revisa la sección de calificaciones."
        ]
    },
    {
        "nombre": "asistencia_nivelacion",
        "utterances": [
            "cuantas faltas puedo tener en nivelacion",
            "como funciona la asistencia en nivelacion",
            "que pasa si falto mucho al nivelatorio",
            "porcentaje de asistencia requerido en nivelacion",
            "cuantas inasistencias me permiten",
            "me pueden reprobar por faltas en nivelacion",
        ],
        "respuestas": [
            "Se requiere mínimo 70-75% de asistencia en nivelación. "
            "Superar el límite de faltas puede resultar en reprobación automática. "
            "Consulta el reglamento en https://admision.ug.edu.ec/nivelacion/."
        ]
    },
    {
        "nombre": "recuperar_contrasena",
        "utterances": [
            "olvide mi contrasena del portal",
            "como recupero mi contrasena de admision",
            "no puedo acceder a mi cuenta",
            "perdi mi clave del portal ug",
            "como cambio mi contrasena",
            "no recuerdo mi clave de admision",
            "me olvide la clave de mi cuenta ug",
        ],
        "respuestas": [
            "Para recuperar tu contraseña: 1) Ve a admision.ug.edu.ec, "
            "2) Clic en '¿Olvidaste tu contraseña?', 3) Ingresa tu cédula o correo, "
            "4) Revisa tu email para las instrucciones."
        ]
    },
    {
        "nombre": "segunda_opcion_carrera",
        "utterances": [
            "puedo poner segunda opcion de carrera",
            "como funciona la segunda opcion en la ug",
            "que pasa si no me aceptan en mi primera opcion",
            "puedo elegir otra carrera si no paso",
            "cuantas opciones de carrera puedo poner",
        ],
        "respuestas": [
            "Puedes seleccionar más de una carrera en el sistema SENESCYT. "
            "Si no obtienes cupo en tu primera opción, el sistema asigna según disponibilidad. "
            "Consulta las reglas en admision.ug.edu.ec y senescyt.gob.ec."
        ]
    },
    {
        "nombre": "informacion_contacto",
        "utterances": [
            "como contacto a admision de la ug",
            "correo de admision de la universidad de guayaquil",
            "donde puedo ir para preguntas de admision",
            "informacion de contacto de la ug",
            "como comunicarme con la universidad",
            "atencion al aspirante universidad guayaquil",
        ],
        "respuestas": [
            "Contacta a Admisión UG por: Portal: admision.ug.edu.ec, "
            "Facebook: Admisión UG, Instagram: Coordinación de Admisión y Nivelación."
        ]
    },
    {
        "nombre": "saludo",
        "utterances": [
            "hola",
            "buenos dias",
            "buenas tardes",
            "buenas noches",
            "hola como estas",
            "buen dia",
            "hey",
        ],
        "respuestas": [
            "¡Hola! Bienvenido al asistente virtual de Admisión UG. "
            "Puedo ayudarte con requisitos, fechas, matrícula, nivelación y más. ¿En qué te ayudo?"
        ]
    },
]

ENTIDADES_CARRERAS = [
    "medicina", "derecho", "ingenieria civil", "ingenieria industrial",
    "ciencia de datos", "psicologia", "odontologia", "enfermeria",
    "arquitectura", "contabilidad", "administracion", "economia",
]

TERMINOS_PROCESO = [
    "curso de nivelacion", "cupo aceptado", "matricula ordinaria",
    "ser bachiller", "senescyt", "nivelatorio", "inscripcion", "aspirante",
]

# ── 4. PREPROCESAMIENTO ─────────────────────────────────────────
STEMMER = SnowballStemmer("spanish")
try:
    STOPWORDS = set(stopwords.words("spanish"))
except Exception:
    STOPWORDS = set()
STOPWORDS.update({"ug", "universidad", "guayaquil", "favor", "gracias",
                   "podria", "puede", "quiero", "saber", "decirme"})


def preprocesar(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip():
        return ""
    # Minúsculas + quitar acentos
    texto = texto.lower().strip()
    nfkd = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Limpiar puntuación
    texto = re.sub(r"[¿?¡!.,;:()\[\]{}'\"/@#$%^&*+=<>~`|\\]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    # Tokenizar → stopwords → stemming
    try:
        tokens = word_tokenize(texto, language="spanish")
    except Exception:
        tokens = texto.split()
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ── 5. EXTRACCIÓN DE ENTIDADES ──────────────────────────────────
def extraer_entidades(texto: str) -> dict:
    try:
        tl = texto.lower()
        cedulas = re.findall(r"\b\d{10}\b", texto)
        fechas = re.findall(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            r"|\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|"
            r"julio|agosto|septiembre|octubre|noviembre|diciembre)\b", texto, re.I)
        carreras = [c for c in ENTIDADES_CARRERAS if c in tl]
        terminos = [t for t in TERMINOS_PROCESO if t in tl]
        return {"cedulas": cedulas, "fechas": fechas, "carreras": carreras, "terminos": terminos}
    except Exception:
        return {"cedulas": [], "fechas": [], "carreras": [], "terminos": []}


# ── 6. MOTOR TF-IDF ─────────────────────────────────────────────
class ChatbotUG:
    UMBRAL = 0.20
    FALLBACKS = [
        "Lo siento, no entendí tu consulta. ¿Puedes reformularla?",
        "No encontré información sobre eso. Intenta preguntar sobre requisitos, "
        "fechas, matrícula o nivelación.",
        "Esa consulta está fuera de mi alcance. Visita https://admision.ug.edu.ec/admision/",
    ]

    def __init__(self):
        self._mapa_respuestas = {}
        self._mapa_idx = []          # índice utterance → intención
        utterances_proc = []

        for intent in INTENCIONES:
            nombre = intent["nombre"]
            self._mapa_respuestas[nombre] = intent["respuestas"]
            for utt in intent["utterances"]:
                self._mapa_idx.append(nombre)
                utterances_proc.append(preprocesar(utt))

        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                    max_df=0.95, sublinear_tf=True)
        self._matriz = self._vec.fit_transform(utterances_proc)

    def responder(self, consulta: str) -> dict:
        try:
            if not consulta or not consulta.strip():
                return {"texto": "Por favor escribe tu pregunta.", "intencion": None,
                        "similitud": 0.0, "fallback": True}

            entidades = extraer_entidades(consulta)
            proc = preprocesar(consulta)
            if not proc:
                return {"texto": random.choice(self.FALLBACKS), "intencion": None,
                        "similitud": 0.0, "fallback": True}

            vec_q = self._vec.transform([proc])
            sims = cosine_similarity(vec_q, self._matriz).flatten()
            idx = int(np.argmax(sims))
            sim = float(sims[idx])

            if sim >= self.UMBRAL:
                intencion = self._mapa_idx[idx]
                respuesta = random.choice(self._mapa_respuestas[intencion])
                if entidades["carreras"]:
                    respuesta += f"\n\n📌 Carrera detectada: {entidades['carreras'][0].capitalize()}."
                return {"texto": respuesta, "intencion": intencion,
                        "similitud": round(sim, 4), "fallback": False}
            else:
                return {"texto": random.choice(self.FALLBACKS), "intencion": None,
                        "similitud": round(sim, 4), "fallback": True}

        except Exception as e:
            return {"texto": "Error procesando tu consulta. Intenta de nuevo.",
                    "intencion": None, "similitud": 0.0, "fallback": True}


# ── 7. EVALUACIÓN AUTOMÁTICA ────────────────────────────────────
PRUEBAS = [
    ("que documentos necesito para la admision",       "preguntar_requisitos_admision"),
    ("cuales son los requisitos para entrar a la ug",  "preguntar_requisitos_admision"),
    ("que papeles me piden para inscribirme",           "preguntar_requisitos_admision"),
    ("cuando son las fechas de inscripcion",           "consultar_fechas_registro"),
    ("hasta cuando puedo registrarme",                 "consultar_fechas_registro"),
    ("cronograma de registro segun cedula",            "consultar_fechas_registro"),
    ("como me registro en la pagina de admision",      "crear_cuenta_portal"),
    ("no puedo crear mi cuenta en el portal",          "crear_cuenta_portal"),
    ("pasos para registrase como aspirante en la ug",  "crear_cuenta_portal"),
    ("cuando empiezan las clases de nivelacion",       "consultar_fechas_nivelacion"),
    ("cual es la fecha del curso nivelatorio",         "consultar_fechas_nivelacion"),
    ("cuanto necesito para pasar nivelacion",          "aprobar_nivelacion"),
    ("como funciona la aprobacion en nivelacion",      "aprobar_nivelacion"),
    ("como saber si me aceptaron en la carrera",       "consultar_cupo_aceptado"),
    ("donde veo si tengo cupo en la universidad",      "consultar_cupo_aceptado"),
    ("como hago la matricula en la ug",                "proceso_matricula"),
    ("cuando son las fechas de matriculacion",         "proceso_matricula"),
    ("cuanto debo sacar en el examen senescyt",        "puntaje_examen_senescyt"),
    ("que puntaje minimo piden en la ug",              "puntaje_examen_senescyt"),
    ("cuantas faltas me permiten en nivelacion",       "asistencia_nivelacion"),
    ("que pasa si falt mucho al nivelatorio",          "asistencia_nivelacion"),
    ("olvide mi clave del portal de admision",         "recuperar_contrasena"),
    ("como cambio mi contrasena de la cuenta ug",      "recuperar_contrasena"),
    ("hola buenos dias",                               "saludo"),
    ("que carreras tiene la ug",                       "carreras_disponibles"),
]


def evaluar(bot: ChatbotUG):
    y_true, y_pred = [], []
    for consulta, esperado in PRUEBAS:
        pred = bot.responder(consulta)["intencion"] or "fallback"
        y_true.append(esperado)
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n" + "="*55)
    print("   EVALUACIÓN DEL AGENTE CONVERSACIONAL UG")
    print("="*55)
    print(f"  Consultas evaluadas : {len(PRUEBAS)}")
    print(f"  Accuracy            : {acc*100:.1f}%")
    print(f"  F1-Macro            : {f1*100:.1f}%")
    correctos = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"  Correctas           : {correctos}/{len(PRUEBAS)}")
    errores = [(PRUEBAS[i][0], t, p) for i,(t,p) in enumerate(zip(y_true,y_pred)) if t!=p]
    if errores:
        print(f"\n  ⚠️  Errores ({len(errores)}):")
        for q, t, p in errores:
            print(f"     '{q}'\n     esperado={t} | predicho={p}")
    print("="*55 + "\n")
    return acc, f1


# ── 8. INTERFAZ GRADIO ──────────────────────────────────────────
def lanzar_gradio(bot: ChatbotUG):
    def chat(mensaje, historial):
        if not mensaje.strip():
            return historial, ""
        resultado = bot.responder(mensaje)
        respuesta = resultado["texto"]
        if not resultado["fallback"] and resultado["similitud"] < 0.35:
            respuesta += f"\n_(confianza baja: {resultado['similitud']:.0%})_"
        historial.append((mensaje, respuesta))
        return historial, ""

    with gr.Blocks(title="Chatbot Admisión UG", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 🎓 Asistente Virtual — Admisión y Nivelación UG
**Universidad de Guayaquil** | PLN Clásico (TF-IDF + Similitud Coseno)

Pregunta sobre requisitos, fechas, matrícula, nivelación, cupos y más.
""")
        chatbot = gr.Chatbot(label="Conversación", height=430)
        with gr.Row():
            txt = gr.Textbox(placeholder="Escribe tu pregunta...",
                             show_label=False, scale=4)
            gr.Button("Enviar", variant="primary", scale=1).click(
                chat, [txt, chatbot], [chatbot, txt])
        txt.submit(chat, [txt, chatbot], [chatbot, txt])

        gr.Examples(
            examples=[
                "¿Cuáles son los requisitos para la admisión?",
                "¿Cuándo son las fechas de registro?",
                "¿Cómo creo mi cuenta en el portal?",
                "¿Qué puntaje necesito para entrar a la UG?",
                "¿Cuántas faltas puedo tener en nivelación?",
                "Olvidé mi contraseña del portal",
            ],
            inputs=txt
        )

    print("🌐 Abriendo interfaz web en http://localhost:7860")
    print("   (Cierra con Ctrl+C cuando termines)\n")
    demo.launch(share=False)


# ── MAIN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🎓 CHATBOT ADMISIÓN Y NIVELACIÓN UG")
    print("="*40)
    print("⏳ Inicializando agente...")
    bot = ChatbotUG()
    print(f"✓ Agente listo con {len(INTENCIONES)} intenciones\n")

    # Evaluación automática al arrancar
    evaluar(bot)

    # Lanzar interfaz web
    lanzar_gradio(bot)

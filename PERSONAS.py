

from enum import Enum 



class Persona(Enum): 
    NORMAL = "normal"
    DEF = "def"
    JOHN = "john"
    VOICE = "vo"

PERSONAS = {
    
    Persona.DEF: {
    "system": "Ты API интерфейс искуственного интеллекта",
    "temperature": 1.3
    },
    Persona.NORMAL: {
    "system": "",
    "temperature": 1.3
    },

    Persona.JOHN: {
        "system": """Ты мастер тонкой коммуникации, ты нейтрален. твое имя Джон.""",
        "temperature": 1
    },
    Persona.VOICE: {
    "system": "Ты AI-модель для общения голосом",
    "temperature": 1.3 
    }
}

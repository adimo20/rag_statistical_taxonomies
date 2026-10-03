from pathlib import Path

def make_path(filename:str):
    return Path(__file__).resolve().parent.parent / "_data" / filename

mapping = {
    "SEA_2021":{
        "classification_system":"SEA", 
        "loader_name":"KLASS_SERVER",
        "data_path":make_path("sea2021.xml"),
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/sea_2021",
        }
    },
    "SEA_2021_ADDED_MISSING_CODES":{
        "classification_system":"SEA", 
        "loader_name":"JSON",
        "data_path":make_path("SEA_ADD_MISSING_CODES.json"), 
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/sea_2021",
        }
    },
    "SEA_2021_ADDED_CONTEXT":{
        "classification_system":"SEA", 
        "loader_name":"JSON", 
        "data_path":make_path("sea_additional_context.json"), 
        "metadata":{
            "url":"https://klassifikationsserver.de/klassService/thyme/variant/sea_2021",
        }
    }
}

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoConfig

import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory della cache Hugging Face
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
HF_HOME = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
CACHE_DIR_EXT = os.path.join("/Volumes", "Hd Esterno 2", "huggingface_cache","hub")
CACHE_DIR = os.path.join("/Volumes","Macintosh HD",  "Users/maxgiu/.cache/huggingface/hub")
# Definisci il percorso della cache dei modelli Llama-3.2-3B_fine

#CACHE_DIR = os.path.join(HF_HOME, "models--")


def get_available_models():
    models = []
    print("Cache_dir",CACHE_DIR)
    print("pathesiste  ",  os.path.exists(CACHE_DIR))

    # Modelli dalla cache di Hugging Face
    if os.path.exists(CACHE_DIR):
        print("Cache_dir",CACHE_DIR)
        cached_models = [f for f in os.listdir(CACHE_DIR) if f.startswith("models--") and os.path.isdir(os.path.join(CACHE_DIR, f))]
        print("Cached_models",cached_models)
        models.extend([(model.replace("models--",""), os.path.join(CACHE_DIR, model)) for model in cached_models])
        print("Models",models)

    localpath_meta3_2 = os.path.join(CACHE_DIR_EXT, "models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062")
    print("Localpath_meta3_2",localpath_meta3_2)
    models.append(("llama 3.2", localpath_meta3_2))
    models.append(("llama 3.1-70B", "meta-llama/Llama-3.1-70B-Instruct"))


    return models

# Funzione per caricare il modello e il tokenizer
@st.cache_resource
def load_model(model_path):
    cache_dir = "/Volumes/Hd\ Esterno\ 2/huggingface_cache/hub"
    cache_dir =os.path.join("/Volumes", "Hd Esterno 2", "huggingface_cache","hub")
    #cache_dir =os.path.join("/Volumes","Macintosh HD",  "Users/maxgiu/.cache/huggingface/hub")
    print("Cache_dir_newwww",cache_dir)
    #model_name = "meta-llama/Llama-3.2-3B"
    #cache_dir = "/Volumes/Hd Esterno 2/huggingface_cache"
    

    # Scarica il file di configurazione forzatamente
    #config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, force_download=True)
    #print("File di configurazione scaricato con successo.")
    # Scarica il tokenizer forzatamente
    #tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, force_download=True)
    #print("Tokenizer scaricato con successo.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGING_FACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=HUGGING_FACE_TOKEN)

    #tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGING_FACE_TOKEN)
    #model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=HUGGING_FACE_TOKEN)
    
    # Imposta il token di padding se non è presente
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cpu")  # Forza il caricamento su CPU
    model.to(device)
    return model, tokenizer, device

# Funzione per generare risposte dal modello
def generate_response(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=10000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Configura l'interfaccia Streamlit
def chat_with_model():
    st.title("Chatbot con Modello Fine-Tuned")
    st.write("Interagisci con il tuo modello personalizzato!")

    # Mostra i modelli disponibili, inclusa la cartella llama_finetuned
    models = get_available_models()
    model_names = [name for name, _ in models]
    selected_model_name = st.selectbox("Seleziona un modello:", model_names)

    # Aggiungi un pulsante per caricare il modello
    if st.button("Carica Modello"):
        # Trova il percorso del modello selezionato
        model_path = next(path for name, path in models if name == selected_model_name)

        # Carica il modello selezionato
        model, tokenizer, device = load_model(model_path)
        st.success(f"Modello '{selected_model_name}' caricato con successo!")

        # Input utente per la chat
        user_input = st.text_input("Scrivi qui il tuo messaggio:")
        if st.button("Invia"):
            if user_input:
                # Genera risposta
                response = generate_response(user_input, model, tokenizer, device)
                print("Risposta del bot:", response)
                st.write("Risposta del bot:", response)
            else:
                st.write("Inserisci un messaggio per ricevere una risposta.")
    else:
        st.write("Seleziona un modello e premi 'Carica Modello' per iniziare.")






def chat_with_model_old():
    st.title("Chatbot con Modello Fine-Tuned")
    st.write("Interagisci con il tuo modello personalizzato!")

    # Mostra i modelli disponibili, inclusa la cartella llama_finetuned
    models = get_available_models()
    model_names = [name for name, _ in models]
    selected_model_name = st.selectbox("Seleziona un modello:", model_names)

    # Trova il percorso del modello selezionato
    model_path = next(path for name, path in models if name == selected_model_name)

    if selected_model_name:
        # Carica il modello selezionato
        model, tokenizer, device = load_model(model_path)

        # Input utente
        user_input = st.text_input("Scrivi qui il tuo messaggio:")
        if st.button("Invia"):
            if user_input:
                # Genera risposta
                response = generate_response(user_input, model, tokenizer, device)
                st.write("Risposta del bot:", response)
            else:
                st.write("Inserisci un messaggio per ricevere una risposta.")

# Esegui l'app
if __name__ == "__main__":
    chat_with_model()

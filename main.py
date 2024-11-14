import argparse
import pdfplumber
import json
import re
import os
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import openai
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Configuration
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def importa_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + ' '
    return text

# Funzione per pulire il testo estratto
def pulisci_testo(testo):
    # Rimuove caratteri speciali, multipli spazi e converte tutto in minuscolo
    testo = re.sub(r'\s+', ' ', testo)  # Rimuove spazi multipli
    testo = re.sub(r'[^\w\s.,?!]', '', testo)  # Rimuove caratteri speciali tranne punteggiatura di base
    return testo.strip()

# Funzione per leggere tutti i PDF in una cartella
def leggi_tutti_i_pdf(cartella):
    testi = []
    for file_name in os.listdir(cartella):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(cartella, file_name)
            print(f"Importando PDF: {file_path}")
            testo_grezzo = importa_pdf(file_path)
            testo_pulito = pulisci_testo(testo_grezzo)
            testi.append(testo_pulito)
    return testi

# Funzione per autenticare Hugging Face
def auth_huggingface(token):
    try:
        login(token)
        print("Autenticazione su Hugging Face riuscita.")
    except Exception as e:
        print(f"Errore durante l'autenticazione su Hugging Face: {e}")

# Funzione per caricare il modello e il tokenizer
def carica_modello(model_name, token):
    try:
        auth_huggingface(token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

        print(f"Modello '{model_name}' e tokenizer caricati con successo.")
        input_text = "Quali sono i benefici della meditazione?"
        inputs = tokenizer(input_text, return_tensors="pt")
        output = model.generate(**inputs, max_length=100)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        print("Testo generato:")
        print(generated_text)
        return model, tokenizer

    except Exception as e:
        print(f"Errore durante il caricamento o l'utilizzo del modello '{model_name}': {e}")
        return None, None
    
def correggi_frase(frase, api_key):
    """
    Utilizza OpenAI per correggere e migliorare la frase.
    
    Args:
        frase (str): La frase da correggere.
        api_key (str): La chiave API di OpenAI.
    
    Returns:
        str: La frase corretta e migliorata.
    """
    openai.api_key = api_key
    try:
        # Effettua una chiamata a OpenAI per correggere e migliorare la frase
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Usa il modello ChatGPT o un modello compatibile con la versione 1.0.0
            messages=[
                {"role": "system", "content": "Sei un assistente virtuale che crea il dataset per delle frasi mindset e mental coach, ti viene data una frase e tu devi realizzare una frase positiva e ispirante."},
                {"role": "user", "content": f"Correggi e migliora questa frase per essere più ispirante e positiva: {frase}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Errore durante la chiamata a OpenAI: {e}")
        return frase
    

# Funzione per creare un dataset strutturato
def crea_dataset_da_cartella(cartella, output_file="fine_tuning_dataset.jsonl"):
    testi = leggi_tutti_i_pdf(cartella)
    api_key = OPENAI_API_KEY
    totale_frasi = sum(len(testo.split('. ')) for testo in testi)
    dataset = []
    frase_numero = 0
    print(f"Totale frasi da elaborare: {totale_frasi}")

    with open(output_file, "a") as f:
        print(f"Creazione del dataset progressiva in {output_file}")
        for testo in testi:
            frasi = testo.split('. ')
            for frase in frasi:
                if frase:
                    frase_numero += 1
                    percentuale_avanzamento = (frase_numero / totale_frasi) * 100
                    print(f"Elaborazione frase {frase_numero}/{totale_frasi} ({percentuale_avanzamento:.2f}%) - {frase}")
                    frase_corretta = correggi_frase(frase.strip(), api_key)
                    print("Frase da OpenAI: " + frase_corretta)
                    item = {
                        "prompt": f"Affermati positivi e di ispirazione: {frase.strip()}",
                        "completion": frase_corretta
                    }
                    f.write(json.dumps(item) + "\n")
                    dataset.append(item)
    print(f"Dataset creato e salvato progressivamente in {output_file}")

def fine_tuning_hugg(model_name, dataset_file, repo_name, output_dir="./llama_finetuned", token=""):
    auth_huggingface(token)
    cache_dir = "/Volumes/Hd Esterno 2/huggingface_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token,cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=dataset_file)

    def tokenize_function(examples):
        # Tokenizza il prompt
        inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    
        # Copia gli input_ids come labels
        inputs["labels"] = inputs["input_ids"].copy()
    
        return inputs
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        push_to_hub=True,
        hub_model_id=repo_name,
        hub_token=token,
        report_to="none",
        fp16=False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        data_collator=data_collator  # Sostituzione del tokenizer
    )


    trainer.train()
    trainer.push_to_hub()
    print("Modello fine-tuned e caricato con successo su Hugging Face.")

def main():
    parser = argparse.ArgumentParser(description="Leggi tutti i file PDF in una cartella e crea un dataset JSONL per il fine-tuning")
    subparsers = parser.add_subparsers(dest="comando")

    parser_crea_dataset = subparsers.add_parser("crea_dataset", help="Crea un dataset da tutti i PDF in una cartella --folder_path e salva il file JSONL --output_file")
    parser_crea_dataset.add_argument("--folder_path", type=str, required=True, help="Percorso della cartella contenente i PDF")
    parser_crea_dataset.add_argument("--output_file", type=str, default="fine_tuning_dataset.jsonl", help="Nome del file di output JSONL")

    parser_fine_tuning = subparsers.add_parser("fine_tuning", help="Esegue il fine-tuning del modello")
    parser_fine_tuning.add_argument("--dataset_file", type=str, required=True, help="Percorso del file dataset per il fine-tuning")
    parser_fine_tuning.add_argument("--model_name", type=str, required=True, help="Nome del modello per Hugging Face")
    parser_fine_tuning.add_argument("--repo_name", type=str, required=True, help="Nome del repository su Hugging Face per caricare il modello")
    parser_fine_tuning.add_argument("--output_dir", type=str, default="./llama_finetuned", help="Cartella di output per il modello fine-tuned")

    parser_carica_modello = subparsers.add_parser("carica_modello", help="Carica e verifica se il modello è stato caricato correttamente")
    parser_carica_modello.add_argument("--model_name", type=str, required=True, help="Nome del modello da caricare")

    args = parser.parse_args()

    if args.comando == "crea_dataset":
        crea_dataset_da_cartella(args.folder_path, args.output_file)
    elif args.comando == "fine_tuning":
        fine_tuning_hugg(
            model_name=args.model_name,
            dataset_file=args.dataset_file,
            repo_name=args.repo_name,
            output_dir=args.output_dir,
            token=HUGGING_FACE_TOKEN
        )
    elif args.comando == "carica_modello":
        carica_modello(args.model_name, HUGGING_FACE_TOKEN)
    else:
        print("Comando non riconosciuto. Usa --help per vedere la lista dei comandi disponibili.")

if __name__ == "__main__":
    main()

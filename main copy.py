import argparse
import pdfplumber
import json
import re
import os
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
import openai
from huggingface_hub import login


from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # requires OpenAI Realtime API Access

# Funzione per importare e leggere il contenuto di un PDF
def carica_modello(model_name):
    try:
        login("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")  # Sostituisci "il_tuo_token" con la tua API key di Hugging Face
        # Carica il tokenizer e il modello dalla Hugging Face Model Hub
        token="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"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

        print(f"Modello '{model_name}' e tokenizer caricati con successo.")

        # Test con un input di esempio
        input_text = "Quali sono i benefici della meditazione?"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generazione del testo
        output = model.generate(**inputs, max_length=100)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        print("Testo generato:")
        print(generated_text)
        return model, tokenizer

    except Exception as e:
        print(f"Errore durante il caricamento o l'utilizzo del modello '{model_name}': {e}")
        return None, None


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

# Funzione per creare un dataset strutturato da una lista di testi puliti
import openai
import json

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


def crea_dataset_da_cartella(cartella, output_file="fine_tuning_dataset.jsonl",):
    """
    Legge tutti i PDF nella cartella, li pulisce e utilizza OpenAI per migliorare le frasi.
    
    Args:
        cartella (str): La cartella contenente i PDF.
        output_file (str): Il nome del file di output in formato JSONL.
        api_key (str): La chiave API di OpenAI.
    """
    # Legge tutti i PDF nella cartella e li pulisce
    testi = leggi_tutti_i_pdf(cartella)
    api_key=OPENAI_API_KEY
    # Calcola il numero totale di frasi
    totale_frasi = sum(len(testo.split('. ')) for testo in testi)

    dataset = []
    frase_numero = 0  # Contatore delle frasi elaborate
    print(f"Totale frasi da elaborare: {totale_frasi}")

    # Apri il file in modalità append, in modo da poter aggiungere dati man mano
    with open(output_file, "a") as f:
        print(f"Creazione del dataset progressiva in {output_file}")
        for testo in testi:
            # Suddivide il testo in frasi per creare input-output per il fine-tuning
            frasi = testo.split('. ')
            for frase in frasi:
                if frase:
                    frase_numero += 1  # Incrementa il contatore
                    # Calcola la percentuale di avanzamento
                    percentuale_avanzamento = (frase_numero / totale_frasi) * 100
                    print(f"Elaborazione frase {frase_numero}/{totale_frasi} ({percentuale_avanzamento:.2f}%) - {frase}")

                    # Corregge la frase utilizzando OpenAI
                    frase_corretta = correggi_frase(frase.strip(), api_key)
                    print("Frase da Open ai" + frase_corretta)

                    prompt = "Affermati positivi e di ispirazione: " + frase.strip()
                    completion = frase_corretta
                    item = {
                        "prompt": prompt,
                        "completion": completion
                    }
                    
                    # Scrivi immediatamente l'item corretto nel file
                    f.write(json.dumps(item) + "\n")
                    
                    # Aggiungi l'item al dataset (opzionale, se desideri tenerlo in memoria)
                    dataset.append(item)

    print(f"Dataset creato e salvato progressivamente in {output_file}")

   

    

def fine_tuning_hugg(model_name, dataset_file, repo_name, output_dir="./llama_finetuned", token=""):
    # Autenticazione su Hugging Face con il token API
    login(token)  # Assicurati di aver sostituito "token" con la tua API key di Hugging Face
    print("argomenti", model_name, dataset_file, repo_name, output_dir, token)

    # Carica il tokenizer e il modello dalla Hugging Face Model Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    tokenizer.pad_token = tokenizer.eos_token

    # Carica il dataset dal file JSON
    dataset = load_dataset("json", data_files=dataset_file)

    # Tokenizza il dataset
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Imposta gli argomenti per il fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        push_to_hub=True,  # Attiva il push su Hugging Face Hub
        hub_model_id=repo_name,  # Specifica il nome del repository su Hugging Face Hub
        hub_token=token,  # Utilizza il token di autenticazione per il push
        report_to="none",  # Disattiva reporting esterni
        fp16=False  # Utilizza la precisione a 16 bit se disponibile
    )

    # Imposta il trainer per il fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer
    )

    # Esegue il fine-tuning e carica il modello su Hugging Face Hub
    trainer.train()
    trainer.push_to_hub()

    print("Modello fine-tuned e caricato con successo su Hugging Face.")


# Funzione per il fine-tuning con Hugging Face local
def fine_tuning_hugg_local(model_name, dataset_file, repo_name, output_dir="./llama_finetuned", token=""):
    # Autenticazione su Hugging Face con il token API
    login(token)  # Assicurati di aver sostituito "token" con la tua API key di Hugging Face
    print("argomenti", model_name, dataset_file, repo_name, output_dir, token)
    # Carica il tokenizer e il modello dalla Hugging Face Model Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    # Carica il dataset dal file JSON
    dataset = load_dataset("json", data_files=dataset_file)

    # Tokenizza il dataset
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Imposta gli argomenti per il fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        push_to_hub=True,  # Attiva il push su Hugging Face Hub
        hub_model_id=repo_name,  # Specifica il nome del repository su Hugging Face Hub
        hub_token=token,  # Utilizza il token di autenticazione per il push
    )

    # Imposta il trainer per il fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer
    )

    # Esegue il fine-tuning e carica il modello su Hugging Face Hub
    trainer.train()
    trainer.push_to_hub()

    print("Modello fine-tuned e caricato con successo su Hugging Face.")
# Funzione per il fine-tuning con OpenAI
def fine_tuning_openai(dataset_file):
    # Carica e prepara il dataset
    with open(dataset_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # Assicurati di avere l'API key di OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Carica il dataset su OpenAI e avvia il fine-tuning
    response = openai.File.create(file=open(dataset_file), purpose="fine-tune")
    file_id = response["id"]

    # Avvia il processo di fine-tuning
    fine_tune_response = openai.FineTune.create(
        training_file=file_id,
        model="davinci"
    )

    fine_tune_id = fine_tune_response["id"]
    print(f"Fine-tuning avviato con OpenAI. ID del lavoro: {fine_tune_id}")

# Funzione per avviare il fine-tuning in base alla piattaforma selezionata

# Funzione principale per gestire i comandi
def main():
    parser = argparse.ArgumentParser(description="Leggi tutti i file PDF in una cartella e crea un dataset JSONL per il fine-tuning")
    subparsers = parser.add_subparsers(dest="comando")

    # Comando per leggere tutti i PDF e creare un dataset
    parser_crea_dataset = subparsers.add_parser("crea_dataset", help="Crea un dataset da tutti i PDF in una cartella --folder_path e salva il file JSONL --output_file")
    parser_crea_dataset.add_argument("--folder_path", type=str, required=True, help="Percorso della cartella contenente i PDF")
    parser_crea_dataset.add_argument("--output_file", type=str, default="fine_tuning_dataset.jsonl", help="Nome del file di output JSONL")

    # Comando per avviare il fine-tuning
    parser_fine_tuning = subparsers.add_parser("fine_tuning", help="Esegue il fine-tuning del modello")
    parser_fine_tuning.add_argument("--dataset_file", type=str, required=True, help="Percorso del file dataset per il fine-tuning")
    parser_fine_tuning.add_argument("--model_name", type=str, help="Nome del modello per Hugging Face (richiesto se si usa Hugg)")
    parser_fine_tuning.add_argument("--output_dir", type=str, default="./llama_finetuned", help="Cartella di output per il modello fine-tuned (solo per Hugging Face)")
    parser_fine_tuning.add_argument("--repo_name", type=str, help="Nome del repository su Hugging Face per caricare il modello (richiesto se si usa Hugg)")
    
    # Comando per caricare e verificare un modello
    parser_carica_modello = subparsers.add_parser("carica_modello", help="Carica e verifica se il modello è stato caricato correttamente")
    parser_carica_modello.add_argument("--model_name", type=str, required=True, help="Nome del modello da caricare")


    args = parser.parse_args()

    # Esegui la funzione corrispondente al comando
    if args.comando == "crea_dataset":
        print(f"Creazione del dataset da cartella: {args.folder_path}")
        crea_dataset_da_cartella(args.folder_path, args.output_file)

    elif args.comando == "fine_tuning":
        print(f"Avvio del fine-tuning su piattaforma: {args.model_name}")
        fine_tuning_hugg(
                model_name=args.model_name,
                dataset_file=args.dataset_file,
                repo_name=args.repo_name,
                output_dir=args.output_dir,
                token="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"
            )
    
    elif args.comando == "carica_modello":
        print(f"Caricamento del modello: {args.model_name}")
        carica_modello(args.model_name)

    else:
        print("Comando non riconosciuto. Usa --help per vedere la lista dei comandi disponibili.")

if __name__ == "__main__":
    main()

import gradio as gr
from gradio import components 
import yaml
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
                          AutoTokenizer, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, pipeline)
import argparse

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def get_prompt(human_prompt, context):
    prompt_template = f"""You are an SQL agent. Given the following natural language question and the SQL table context, please formulate an accurate and efficient SQL query that would correctly answer the question. Do not deviate from the fact that the answer always has to be an SQL Query. Use all the required joins and other syntaxes.

Natural Language Question: {human_prompt}
SQL Table Context: {context}

Formulate SQL Query:"""
    return prompt_template

def get_llm_response(prompt, context):
    raw_output = pipe(get_prompt(prompt, context))
    sql_query = raw_output[0]['generated_text'].split('Formulate SQL Query:')[-1].strip()
    return sql_query

def generate_sql_response(question, context):
    return get_llm_response(question, context)

if __name__ == "__main__":
    # Read the YAML configuration file (you should specify your own path)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()

    config = read_yaml_file(args.config_path)

    # Load model
    print("Load model")
    model_path = f"{config['model_output_dir']}/{config['model_name']}"
    if "model_family" in config and config["model_family"] == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)

    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15
    )

    interface = gr.Interface(
        fn=generate_sql_response, 
        inputs=[
            components.Textbox(lines=2, label="Enter your question"),
            components.Textbox(lines=4, label="Enter SQL Table Context")
        ], 
        outputs=components.Textbox(label="SQL Query"),
        live=True
    )
    
    interface.launch()


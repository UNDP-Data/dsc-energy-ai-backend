# standard library
import os
import json
import re

# web
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# wrangling
import numpy as np
import pandas as pd
import faiss
import spacy
import openai
import pandasai
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI

from langchain.indexes.graph import NetworkxEntityGraph
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate

# utils
from dotenv import load_dotenv
load_dotenv()

# visualisation
import matplotlib
matplotlib.use('agg')

# local
from src.json_image import image_to_json


app = Flask(__name__)
CORS(app)

# cache_folder = '/cache'
# cache_db_file = os.path.join(cache_folder, 'cache_db.db')
# cache_db_wal_file = os.path.join(cache_folder, 'cache_db.db.wal')
# if os.path.exists(cache_db_file):
#     os.remove(cache_db_file)
# if os.path.exists(cache_db_wal_file):
#     os.remove(cache_db_wal_file)

deployment_model=os.getenv('OPENAI_DEPLOYMENT_MODEL')
llm = AzureOpenAI(
    deployment_name=deployment_model,
    api_token=os.getenv('OPENAI_API_KEY'),
    api_base=os.g
    etenv('OPENAI_API_BASE'),
    api_version=os.getenv('OPENAI_API_VERSION'),
    # is_chat_model=True,
)

sheets = pd.read_excel('Moonshot Tracker Results.xlsx', sheet_name=None)

openai.api_type = os.getenv('OPENAI_API_TYPE')  
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_key = os.getenv('OPENAI_API_KEY')

nlp = spacy.load("en_core_web_sm")
df = pd.read_pickle('models/df_embed_EN.pkl')

def find_mentioned_countries(text):
    doc = nlp(text)
    countries = set()
    
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE stands for "Geopolitical Entity"
            countries.add(ent.text)
    return list(countries)

def filter_country(user_query):
    mentioned_countries = find_mentioned_countries(user_query)
    if mentioned_countries:
        country = mentioned_countries[0]
    # Proceed with further processing or handling the country variable
    else:
        country = ""    
    # print(country)
    return df[df['Country Name'] == country]

def search_embeddings(user_query):
    df_filtered = filter_country(user_query)
    length = len(df_filtered.head())
    filtered_embeddings_arrays = np.array(list(df_filtered['Embedding']))
    index = faiss.IndexFlatIP(filtered_embeddings_arrays.shape[1]) 
    index.add(filtered_embeddings_arrays)
    
    user_query_embedding = openai.Embedding.create( 
        input=user_query, engine=os.getenv('USER_QUERY_EMBEDDING_ENGINE')  
    )["data"][0]["embedding"]
    
    k = min(5, length)
    distances, indices = index.search(np.array([user_query_embedding]), k)
    return df_filtered, distances, indices

def get_answer(user_question, content):
    system_prompt = "You are a system that answers user questions based on excerpts from PDF documents provided for context. Only answer if the answer can be found in the provided context. Do not make up the answer; if you cannot find the answer, say so."
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question},
        {'role': 'user', 'content': content},
    ]
    
    response = openai.ChatCompletion.create(
        engine=deployment_model,
        messages=messages,
        temperature=0.2,
        # max_tokens=2000
    )
    return response

def response_generating_KG_Model(user_query):
    #Load the KG Model
    loaded_graph = NetworkxEntityGraph.from_gml("models/moonshot_AI_graph_model_v1.gml")
    prompt =  "Use the following knowledge triplets to answer the question at the end. If you don't know the answer, look out for potential factors in the knowledge triplets else just say I don't know based on my knowledge base, don't try to make up an answer. If a term like a Continent is used e.g Africa, Asia, replace the continent with all african countries available in the knowledge triplets. E.g Nigeria, South Africa and Egypt are under Africa. In your answer, Always refer to knowledge triplets as knowledge base.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"
    prompt_entity="Extract all entities from the following text. As a guideline, a proper noun is generally capitalized. You should definitely extract all names,places, Dates and Times, Numbers, Organizations, Products and Brands, Events, Roles and Positions, Keywords and Topics, Email Addresses and URLs, References to External Entities, Emotional Tone, Quantities and Units, Codes and Identifiers, Languages, Social Media Handles, Currencies..\n\nReturn the output as a single comma-separated list, or NONE if there is nothing of note to return.\n\nEXAMPLE\ni'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.\nOutput: Langchain\nEND OF EXAMPLE\n\nEXAMPLE\ni'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I'm working with Sam.\nOutput: Langchain, Sam\nEND OF EXAMPLE\n\nBegin!\n\n{input}\nOutput:"
    chain = GraphQAChain.from_llm(AzureChatOpenAI(temperature=0, deployment_name= os.getenv('OPENAI_DEPLOYMENT_MODEL')), graph=loaded_graph, verbose=False,
    qa_prompt=PromptTemplate(input_variables=['context', 'question'], template=prompt),
    entity_prompt=PromptTemplate(input_variables=['input'], template=prompt_entity)
    )
    response = chain.run(user_query) 
    return response


def response_generating(user_query):
    df, distances, indices = search_embeddings(user_query)
    dis = distances[0][::-1]
    ind = indices[0][::-1]
    
    for i in range(len(dis)):
        content = df.iloc[ind[i]]['content_cut']
        print("Searching document {} ({})...".format(df.iloc[ind[i]]['Document Title'], df.iloc[ind[i]]['Link']))
        response = get_answer(user_query, content)
        answer = response['choices'][0]['message']['content']
        
        not_found_phrases = ['not mention', 'not mentioned', 'I did not find', 'not found', 'no information', 'not contain', 'cannot be found', 'no mention']
        
        if any(phrase.lower() in answer.lower() for phrase in not_found_phrases):
            print('Answer not found in this document')
            continue
        else:
            return answer


# Function to check API key
def require_api_key(api_key):
    valid_keys = [os.getenv('API_ACCESS_KEY') ]  # Replace with your valid API keys
    return api_key in valid_keys

# Apply the before_request decorator to all routes
@app.before_request
def check_api_key():
    # if request.endpoint != 'static_access':
    api_key = request.headers.get('API-Key')
    if api_key != os.getenv('API_ACCESS_KEY'):
        return jsonify({'error': 'Unauthorized access'}), 401

@app.route('/header', methods = ['GET'])
@cross_origin() 
def get_header():
    table = request.get_json()['table_name']
    header = sheets[table].head()
    return header.to_json()


@app.route('/pandasai', methods = ['POST'])
@cross_origin() 
def send_promt_pandasai():

    table_name, prompt_pansasai = request.get_json()['table_name'], request.get_json()['prompt']
    
    lower_case = prompt_pansasai.lower()
    pattern = r"\bbudgets?\b" 
    if re.search(pattern, lower_case):
        df = sheets['Projects']
    else:
        df = sheets['Outputs']

    smart_df = SmartDataframe(df, config={"llm": llm})
    output = smart_df.chat(prompt_pansasai)
    image_path = "exports/charts/temp_chart.png"

    if output is None:
        return image_to_json(image_path)
    elif isinstance(output, pandasai.smart_dataframe.SmartDataframe):
        return pd.DataFrame(output, columns=output.columns).to_json()
    elif isinstance(output, pd.DataFrame) or isinstance(output, pd.Series):
        return output.to_json()
    else:
        return json.dumps(output)



@app.route('/llm', methods = ['POST'])
@cross_origin() 
def send_promt_llm():
    try: 
        promt_llm = request.get_json()['query']
        try:
            answer_search_embeddings = response_generating(promt_llm)
        except Exception as e:
        # Handle exceptions from response_generating function
            answer_search_embeddings = ""  # Set to an appropriate empty value
        
        answer_kg_model = response_generating_KG_Model(promt_llm)
        return jsonify({
                        'status':'success',
                        'message':'Matched result successfully',
                        'answers': [
                           {
                               'source': 'search_embeddings',
                                'value': answer_search_embeddings,
                                'model_version': '1.0.0'
                           },
                                                      {
                               'source': 'knowledge_graph_model',
                                'value': answer_kg_model,
                                'model_version': '1.0.0' 
                           }  
                            ], 
                        'entities': [],
                        'prompts': []
                         })
    except Exception as e:

        print(e)

        # I did not find anything from the existing documents
        return jsonify(
                     {
                        'status':'failed',
                        'message': 'an error occured',
                        'answers': [], 
                        'entities': [],
                        'prompts': []
                         })

    

if __name__ == "__main__":
    app.run()
    

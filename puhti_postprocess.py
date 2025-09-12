import pandas as pd
import ollama
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler
logger = logging.getLogger(__name__)
logging.basicConfig(handlers=[RotatingFileHandler('postprocess.log', encoding='utf-8', maxBytes=1000000, backupCount=5)], level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Sentiment(BaseModel):
    topics: list[str]
    entities: list[str]
    positive: list[str]
    neutral: list[str]
    negative: list[str]


def get_system_prompt():
    # all_entities to strinf

    system_prompt = '''### **System Prompt**

    **Role**:
    - You are presented a previously generated analysis of a political video. Your task is to extract lists of elements from the analysis.
    - Extract entities, sentiments and topics from an analysis of a video. Extract the names of the entities and topics mentioned in the analysis. Provide a simple list of the names of the entities, sentiments and topics. 
    - Present the lists in a structured JSON format. If no entities or topics are found, return an empty list.

    ---

    **Tasks**:
    1. **Extract Topics**:
        - Extract topics mentioned in the analysis.  
        - Provide only the name of each topic.
        - Present the topics in a structured JSON format.
        - If no topics are found, return an empty list.  
        - Try to avoid redundancy in the topics extracted.
        - Merge the topics that are similar or related.
        - If no topics are found, return an empty list.

    2. **Extract Entities**:
        - Extract entities mentioned in the analysis.  
        - Provide only the name of each entity.
        - Present the entities in a structured JSON format.
        - If no entities are found, return an empty list.
        - Try to avoid redundancy in the entities extracted.
        - Merge the entities that are similar or related.
        - If no entities are found, return an empty list.

    3. **Determine Sentiments**:
        - Extract sentiments listed in the text and determine if they are positive, negative, or neutral.
        - Determine the target of each sentiment.
        - Create lists of the targets of positive, negative and neutral sentiments.
        - Try to avoid redundancy in the targets extracted.
        - Merge the targets that are similar or related.
        - If no sentiments are found, return an empty list.

    ---

    **Formatting Rules**:  
    - Always output your responses as **valid JSON objects**.
    - Adhere to the following format exactly:

    ```json
    {
        "topics": ["name of topic", "name of topic"],
        "entities": ["name of entity", "name of entity"],
        "positive": ["name of target", "name of target"],
        "neutral": ["name of target", "name of target"],
        "negative": ["name of target", "name of target"],
    }
    ```

    ---

    **Additional Guidelines**:  
    1. Always ensure the JSON is **well-formed and valid**.  
    2. If a category (e.g., topics, entities) has no data, use an empty array (`[]`).  
    3. Only provide the name of the entity or topic.
    4. Respond only with valid JSON. Do not write an introduction or summary.
    '''
    return system_prompt



def get_response(user_prompt, system_prompt):
    llama_response = {}
    options = {"repeat_last_n": 64,
               "repeat_penalty": 1.1,
               "num_ctx": 4096,
               "top_p": 0.9,
               "top_k": 40,
               "min_p": 0.0,
               "temperature": 0.0,
               "num_predict": 2048}
    try:
        # llama3.3:70b or gemma3:27b 
        response = ollama.chat(model="gemma3:27b", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], options=options, format=Sentiment.model_json_schema())
        llama_response = response['message']['content']
        print(llama_response)
        logger.debug(llama_response)
        llama_response = Sentiment.model_validate_json(llama_response)
        print(llama_response)
        #llama_response = json.loads(llama_response)
        print(llama_response)
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")
    return llama_response


def analyze_responses():
    filename = f'ep24_{country}.csv'
    df = pd.read_csv(filename)
    df["entities"] = ""
    df["topics"] = ""
    df["positive"] = ""
    df["neutral"] = ""
    df["negative"] = ""
    for index, row in df.iterrows():
        print(f'Processing row {index}')
        entity_text = ""
        topic_text = ""
        positive_text = ""
        neutral_text = ""
        negative_text = ""
        try:
            summary_analysis = row["summary_analysis"]            
            user_prompt = row["summary_analysis"]
            system_prompt = get_system_prompt()
            response = get_response(user_prompt, system_prompt)
            # Print response
            print(response)
            # Get the entities, topics, us, them, positive, neutral, negative from Sentiment object
            entities = response.entities
            topics = response.topics
            positive = response.positive
            neutral = response.neutral
            negative = response.negative
            # Make all unique
            if len(entities) > 0:
                entity_text = ', '.join([entity for entity in entities])
                print(f'Entities: {entity_text}')
            if len(topics) > 0:
                topic_text = ', '.join([topic for topic in topics])
                print(f'Topics: {topic_text}')
            if len(neutral) > 0:
                neutral_text = ', '.join([target for target in neutral])
                print(f'Neutral: {neutral_text}')
            if len(positive) > 0:
                positive_text = ', '.join([target for target in positive])
                print(f'Positice: {positive_text}')
            if len(negative) > 0:
                negative_text = ', '.join([target for target in negative])
                print(f'Topics: {negative_text}')
            df.at[index, "entities"] = str(entity_text)
            df.at[index, "topics"] = str(topic_text)
            df.at[index, "positive"] = str(positive_text)
            df.at[index, "neutral"] = str(neutral_text)
            df.at[index, "negative"] = str(negative_text)
            print(f'Processed row {index}')
        except Exception as e:
            print(f'Error processing row {index}: {e}')
            logger.error(f'Error processing row {index}: {e}')
        try:
            # Dataframe to string
            df = df.astype(str)
            new_filename = f'finland_mobile_fixed.csv'
            df.to_csv(new_filename, index=False)
        except Exception as e:
            print(f'Error saving dataframe: {e}')
            logger.error(f'Error saving dataframe: {e}')
    # Dataframe to string
    df = df.astype(str)
    new_filename = f'ep24_{country}.csv'
    df.to_csv(new_filename, index=False)


countries = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr']
analyze_responses()

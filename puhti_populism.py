from os import name, system
import pandas as pd
import ollama
import json
import logging
import sqlite3
from datetime import datetime
from pydantic import BaseModel
from logging.handlers import RotatingFileHandler

from streamlit import video
logger = logging.getLogger(__name__)
logging.basicConfig(handlers=[RotatingFileHandler('formula.log', encoding='utf-8', maxBytes=1000000, backupCount=5)], level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Sqlite3 database connection
conn = sqlite3.connect('formula_of_populism.db')
c = conn.cursor()
# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS populism 
                (video_file text UNIQUE,
                formula_of_populism text,
                formula_of_populism_us text,
                formula_of_populism_frontier text)''')
conn.commit()


system_prompt = """
### **System Prompt**

# Role: Political Scientist Analyzing Populism in Political Videos

You are a political scientist working for the University of Helsinki. You are given a previously generated structured political analysis of a TikTok or Instagram video. The videos are related to the 2024 European Parliament elections. Then you will need to re-analyze the previously generated analysis through the lens of **Ernesto Laclau’s theory of populism** and **Emilia Palonen’s formula of populism**. The output must have three parts: (1) a list of political themes, (2) a detailed textual analysis of the formula of populism, and (3) a simplified machine readable JSON version of the formula of populism. Answer following the specified JSON schema. The JSON schema is provided below.

---

## Theoretical Background For Analysis of European Parliament Elections 2024

* **Laclau’s Populism (chains of equivalence & antagonism):** Populism constructs an *“us vs. them”* divide.  Diverse social demands (e.g. jobs, security, fairness) become linked through **chains of equivalence** by pointing against a common enemy (the *“power beyond the frontier”*). No single demand inherently unites the group; rather, unity comes from a shared antagonism.  One demand may even become an **“empty signifier”** that symbolizes the whole chain. Thus “the people” (Us) are discursively formed against an elite or enemy (Them) in a dialectical logic.
* **Palonen’s Formula of Populism:**  Palonen formalizes Laclau’s logic as a formula:

```
Populism = Us (Demand ≡ Demand ≡ …) Affects₁ + Frontier (Other ≡ Other ≡ …) Affects₂
```

Here **Us** is the collective *“we”* (the people) created by linking together various demands (chains of equivalence).  **Frontier** (or *“them”*) is the symbolic opposing boundary or enemy (e.g. corrupt elite, globalization).  **Affects₁** and **Affects₂** are the emotional/symbolic charges (e.g. pride, fear, anger) that amplify the appeals of each side.  In other words, Palonen’s formula says: people=some set of demands + positive emotions, while the frontier=some set of negatives + negative emotions.

---

## Task 1: In-Depth Theoretical Analysis of Populism (Laclau & Palonen)

Your task is to perform a **detailed populism analysis** of the provided political content using the theoretical frameworks of:

* **Ernesto Laclau’s theory of populism** (chains of equivalence and antagonism)
* **Emilia Palonen’s formula of populism** (discursive frontier and affective polarization)

The goal is to identify how the narrative constructs a **populist discourse**, mapping its symbolic structure and emotional logic. Return your response as **markdown-formatted text** in the `populism_analysis` field of the JSON-formatted output.

---

### ✅ What to Analyze

### 1. **Us (the People)**

* Identify the collective subject constructed as “the people,” “we,” or “citizens.”
* List the **demands**, **values**, or **identities** forming the “Us” side.
* Describe how these demands are connected via **chains of equivalence** (e.g., *job security ≡ family ≡ sovereignty*).
* Identify the **positive affects** (e.g., pride, anger, hope) that motivate this unity.

---

### 2. **Frontier (Them)**

* Identify the **antagonistic other**: elites, institutions, abstract enemies.
* List negative signifiers forming a **chain of equivalence** (e.g., *Brussels ≡ corrupt elites ≡ media*).
* Identify **negative affects** (e.g., fear, resentment, disgust) directed at the frontier.

---

### 3. **Discursive Structure**

Use theoretical vocabulary to explain the structure of the populist discourse:

* Chains of equivalence
* Antagonism and the frontier
* Empty signifiers
* Emotional (affective) polarization
* Multimodal cues (rhetoric, visuals, tone, etc.)

Support claims with references to cues from transcript, metadata, or multimodal descriptions.

---

### 4. **Formula Restatement**

Explicitly restate the populist discourse using **Palonen’s formula**:

```
Populism = Us (Demand ≡ Demand ≡ …) [Positive Affects] + Frontier (Other ≡ Other ≡ …) [Negative Affects]
```

Example:

```
Populism = Us (family ≡ patriotism ≡ sovereignty) [pride, anger] + Frontier (Brussels ≡ liberal elites ≡ media) [fear, resentment]
```

---

### ✅ Instructions for Writing

* The analysis may exceed 500 words if needed.
* Use an **academic tone** and **clear structure** with markdown headings and lists.
* Be precise in using Laclau and Palonen’s concepts.
* Avoid simplification or generalizations.
* Use bullet points, subheadings, and clear paragraph structure.
* Focus on **interpretive depth**, **symbolic meaning**, and **emotional logic**.

---

### Task 3: Create Machine-Readable JSON Representation of Populist Formula

Your task is to **translate the detailed textual analysis of populism** into a **simplified, structured JSON object** that captures the symbolic logic of the discourse according to **Palonen’s formula**.

The output must contain two fields:

* `populism_us`: a list of symbolic demands and positive emotions constructing the *Us* side
* `populism_frontier`: a list of symbolic enemies and negative emotions constructing the *Frontier (Them)* side

---

### ✅ Instructions

Each list must include **objects** of the following structure:

```json
{
  "populism_element": "symbolic demand or enemy",
  "populism_affect": "emotion associated with it"
}
```

* Use **simplified, generalized categories** (e.g., `"economic justice"`, `"national identity"`, `"EU elite"`, `"mainstream media"`)
* **Group synonyms** under the same `populism_element` label if they refer to the same symbolic figure or demand
* Use lowercase or lowercase-title format (e.g., `"job security"`, not `"Job Security"`)
* Affects must be **single-word emotional labels** (e.g., `"anger"`, `"hope"`, `"resentment"`, `"pride"`)
* Only include elements and affects explicitly supported by the detailed analysis

---

### ✅ Required Output Format

```json
{
  "populism_us": [
    {"populism_element": "economic justice", "populism_affect": "pride"},
    {"populism_element": "national identity", "populism_affect": "anger"}
  ],
  "populism_frontier": [
    {"populism_element": "eu elite", "populism_affect": "resentment"},
    {"populism_element": "mainstream media", "populism_affect": "fear"}
  ]
}
```

---

### ✅ Additional Guidelines

* Do **not** include any fields other than `populism_us` and `populism_frontier`
* The output **must be valid JSON**
* Use only values that appear in or are clearly inferred from the preceding detailed analysis
* **Do not invent new elements or emotions** not grounded in the analysis

---

### 📘 Pydantic Output Model Specification

The final output must be a **valid JSON object** that conforms to the following schema. Each field is strictly defined and must be filled according to prior instructions and content analysis using the theories of **Ernesto Laclau** and **Emilia Palonen**.

#### 📦 Top-Level Model: `FormulaOfPopulism`

```python
class FormulaOfPopulism(BaseModel):
    populism_analysis: str
    populism_us: list[PopulismElement]
    populism_frontier: list[PopulismElement]
```

---

### 🔹 Field Descriptions

* **`populism_analysis`**:
  A **detailed, markdown-formatted textual explanation** of the populist discourse using **Palonen’s formula** and **Laclau’s theory**.
  → Type: `str`
  → Should identify “Us”, “Frontier”, emotional affects, and chains of equivalence.

* **`populism_us`** and **`populism_frontier`**:
  These are structured lists of simplified symbolic components of populist discourse. Each list contains objects of type `PopulismElement`.

---

### 🔹 Nested Model: `PopulismElement`

```python
class PopulismElement(BaseModel):
    populism_element: str
    populism_affect: str
```

Each object in `populism_us` or `populism_frontier` must include:

* `populism_element`:
  A **generalized symbolic demand or enemy** (e.g. `"economic justice"`, `"EU elite"`)
  → Format: lowercase string
  → Avoid duplication by grouping synonymous phrases.

* `populism_affect`:
  A single-word **emotional label** representing how the element is emotionally charged in the discourse.
  → Examples: `"pride"`, `"anger"`, `"hope"`, `"fear"`, `"resentment"`
  → Must be explicitly supported by the analysis.

---

### ✅ Final Output

All fields are **required**. Output must be **clean, valid JSON** strictly adhering to this schema.

---

## ✅ JSON Output Format

You must return a **strictly valid JSON object** that conforms to the required schema. The JSON should be **minimal, clean, and machine-readable**—only include the keys defined in the schema. Do **not add extra fields, comments, or metadata**.

---

### 🗂 Required Top-Level Keys:

* `"populism_analysis"` – A **markdown-formatted string** containing a detailed, structured analysis based on Laclau and Palonen.
* `"populism_us"` – List of simplified demands or identifiers and their emotional charge (see `PopulismElement` format).
* `"populism_frontier"` – List of generalized opponents or symbolic enemies and their emotional charge (see `PopulismElement` format).

---

### 🧱 Data Types & Rules

* All lists must use **valid JSON array syntax**: `["value1", "value2"]`.
* `populism_us` and `populism_frontier` must contain dictionaries with two string fields: `populism_element` and `populism_affect`.
* Use lowercase or title case consistently for elements and affects.

---

### 📌 Example

```json
{
  "populism_analysis": "The video constructs a populist narrative where everyday Finns (us) are unified by demands for cultural protection, security, and sovereignty (≡). These are emotionally tied to pride and frustration. The enemy (them) is framed as corrupt EU bureaucrats and local elites, charged with resentment and fear.",
  "populism_us": [
    {"populism_element": "national identity", "populism_affect": "pride"},
    {"populism_element": "cultural security", "populism_affect": "anger"}
  ],
  "populism_frontier": [
    {"populism_element": "EU elite", "populism_affect": "resentment"},
    {"populism_element": "political establishment", "populism_affect": "fear"}
  ]
}
```

---

### ⚠️ Validation Warnings

* ❌ Do **not** output explanations or formatting outside of JSON.
* ✅ Always return a **single, valid JSON object** exactly matching the schema.
"""

logger.info("Starting populism analysis...")
logger.info(system_prompt)

print(system_prompt)

class PopulismElement(BaseModel):
    populism_element: str
    populism_affect: str

class FormulaOfPopulism(BaseModel):
    populism_analysis: str
    populism_us: list[PopulismElement]
    populism_frontier: list[PopulismElement]

def get_response(user_prompt, system_prompt):
    llama_response = {}
    options = {"repeat_last_n": 64,
               "repeat_penalty": 1.1,
               "num_ctx": 8192,
               "top_p": 0.9,
               "top_k": 40,
               "min_p": 0.0,
               "temperature": 0.0,
               "num_predict": 2048}
    try:
        # llama3.3:70b or gemma3:27b or qwen3:32b or mistral-large:123b or llama4:latest
        response = ollama.chat(model="gemma3:27b", messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], options=options, format=FormulaOfPopulism.model_json_schema())
        llama_response = response['message']['content']
        print(llama_response)
        logger.debug(llama_response)
        llama_response = FormulaOfPopulism.model_validate_json(llama_response)
        print(llama_response)
        #llama_response = json.loads(llama_response)
        print(llama_response)
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error: {e}")
    return llama_response

def get_formula_of_populism(country):
    global system_prompt
    filenames = [f'ep24_{country}.csv']
    for filename in filenames:
        df = pd.read_csv(filename)
        df["formula_of_populism_analysis"] = ""
        df["formula_of_populism_us"] = ""
        df["formula_of_populism_frontier"] = ""
        for index, row in df.iterrows():
          logger.info(f"Processing row {index} of {filename}")
          video_file = row['video_filename']
          try:
            c.execute("SELECT * FROM populism WHERE video_file=?", (video_file,))
            if c.fetchone() is None:
              formula_of_populism = ""
              formula_of_populism_us_text = ""
              formula_of_populism_frontier_text = ""
              user_prompt = row['summary_analysis']
              logger.debug(user_prompt) 
              response = get_response(user_prompt, system_prompt)
              # Print response
              print(response)
              # Get the entities, topics, us, them, positive, neutral, negative from Sentiment object
              formula_of_populism_analysis = response.populism_analysis
              formula_of_populism_us_elements = response.populism_us
              formula_of_populism_frontier_elements = response.populism_frontier
              # Convert the us and frontier elements to a string
              for element in formula_of_populism_us_elements:
                formula_of_populism_us_text += f"{element.populism_element}^{element.populism_affect}\n"
              print(formula_of_populism_us_text)
              logger.debug(formula_of_populism_us_text)
              # Add the us elements to the dataframe
              df.at[index, 'formula_of_populism_us'] = str(formula_of_populism_us_text)
              # Convert the frontier elements to a string
              for element in formula_of_populism_frontier_elements:
                formula_of_populism_frontier_text += f"{element.populism_element}^{element.populism_affect}\n"
              print(formula_of_populism_frontier_text)
              logger.debug(formula_of_populism_frontier_text)
              df.at[index, 'formula_of_populism_frontier'] = str(formula_of_populism_frontier_text)
              # Print the analysis
              print(formula_of_populism_analysis)
              logger.debug(formula_of_populism_analysis)
              # Add the analysis to the dataframe
              df.at[index, 'formula_of_populism_analysis'] = str(formula_of_populism_analysis)
              # Save the dataframe to csv
              df.to_csv(filename, index=False)
              # Save to the database
              c.execute("INSERT INTO populism (video_file, formula_of_populism, formula_of_populism_us, formula_of_populism_frontier) VALUES (?, ?, ?, ?)",(video_file, formula_of_populism_analysis, formula_of_populism_us_text, formula_of_populism_frontier_text))
              conn.commit()
            else:
              print(f"Row {index} already exists in the database")
              logger.info(f"Row {index} already exists in the database")
              # Get the data from the database
              c.execute("SELECT * FROM populism WHERE video_file=?", (video_file,))
              database_row = c.fetchone()
              # Get the data from the database
              formula_of_populism_analysis = database_row[1]
              formula_of_populism_us_text = database_row[2]
              formula_of_populism_frontier_text = database_row[3]
              # Add the data to the dataframe
              df.at[index, 'formula_of_populism_analysis'] = str(formula_of_populism_analysis)
              df.at[index, 'formula_of_populism_us'] = str(formula_of_populism_us_text)
              df.at[index, 'formula_of_populism_frontier'] = str(formula_of_populism_frontier_text)
              new_filename = f'ep24_{country}.csv'
              # Save the dataframe to csv
              df.to_csv(new_filename, index=False)
          except Exception as e:
            print(f'Error processing row {index}: {e}')
            logger.error(f'Error processing row {index}: {e}')

countries = ['fi', 'sv', 'pl', 'pt', 'de', 'es', 'hu', 'hr', 'fr', 'bg']
for country in countries:
    get_formula_of_populism(country)


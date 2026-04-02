
import pandas as pd
from openai import OpenAI
import json
# Load the data
df = pd.read_csv("data/transcriptions.csv")
df.head()
import json
from openai import OpenAI

client = OpenAI(api_key="<OPENAI_API_TOKEN>")

function_definition = [
    {
        "type": "function",  
        "function": {
            "name": "extract_trans_info", 
            "description": "Extract age, medical specialty and treatment info from input text",  
            "parameters": {
                "type": "object", 
                "properties": {
                    "age": {          
                        "type": "string",
                        "description": "Age"
                    },
                    "treatment": {     
                        "type": "string",
                        "description": "recommended treatment"
                    }
                },
                "required": ["age", "treatment"]  
            }
        }
    }
]

messages = []

# Prepare columns in DataFrame
df_structured = pd.DataFrame()
df_structured["transcription"]=df["transcription"]
df_structured["age"] = ""
df_structured["treatment"] = ""
df_structured["ICD_code"] = ""

for index, row in df_structured.iterrows():
    messages = [{
        "role": "user",
        "content": row['transcription']
    }]

    response = client.chat.completions.create(
        model="gpt-4o-mini",     
        messages=messages, 
        tools=function_definition
    )

    tool_call = response.choices[0].message.tool_calls[0]
    arguments_json = tool_call.function.arguments    
    info = json.loads(arguments_json) 
    
    df_structured.at[index, "age"] = info["age"]
    df_structured.at[index, "treatment"] = info["treatment"]

  
    icd_messages = [
        {"role": "system", "content": "find the ICF codes based on treatment info"},
        {"role": "user", "content": info["treatment"]}
    ]
    response = client.chat.completions.create(    
        model="gpt-4o-mini",         
        messages=icd_messages
    )
    df_structured.at[index, "ICD_code"] = response.choices[0].message.content

print(df_structured.head())
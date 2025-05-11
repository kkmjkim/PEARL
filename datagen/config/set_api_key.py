import os
import json

def set_openai_api(api_key_name):
    personal_info = json.load(open("datagen/config/personal_info.json", "r"))
    assert api_key_name in personal_info
    os.environ["OPENAI_API_KEY"] = personal_info[api_key_name]["api_key"]
    if "org_id" in personal_info[api_key_name]:
        os.environ["OPENAI_ORGANIZATION"] = personal_info[api_key_name]["org_id"]
    print(f"Set OpenAI API Key and Organization of {api_key_name}.") 
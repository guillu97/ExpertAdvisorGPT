from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

# Sécurise la base URL au cas où ton env traîne une valeur incorrecte
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.openai.com/v1"
)
print(client.api_key)
res = client.responses.create(
    model="gpt-5-nano",
    input="Dis bonjour en une phrase."
)
print(res.output_text)
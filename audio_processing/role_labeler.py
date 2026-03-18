import os
from openai import OpenAI

def create_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def relabel_speakers_with_llm(client: OpenAI, llm_model: str, transcript: str) -> str:
    print("Identifying Agent vs Customer...")

    prompt = ("The following call transcript contains two speakers: Speaker_0 and Speaker_1.\n\n"
              "Identify wich one is the Agent and which one is the Customer. \n"
              "Rewrite the transcript replacing Speaker_0 and Speaker_1.\n\n"
              f"Transcript:\n{transcript}")
    
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": "You analyze call center conversations."},
                  {"role": "user", "content": prompt},],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content.")
    return content
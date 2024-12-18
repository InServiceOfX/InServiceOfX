def create_user_message(prompt: str) -> dict:
    return {"role": "user", "content": prompt}

def create_system_message(prompt: str) -> dict:
    return {"role": "system", "content": prompt}

from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def run():
    result = subprocess.check_output(["python", "inference.py"]).decode()
    return {"output": result}
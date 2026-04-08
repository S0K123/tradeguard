from fastapi import FastAPI
import asyncio
from inference import main

app = FastAPI()

@app.get("/")
def home():
    return {"status": "TradeGuard is running"}

@app.post("/reset")
async def reset():
    try:
        await main()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
# server/app.py
from fastapi import FastAPI
import asyncio
from inference import main  # your existing inference.py function

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

if __name__ == "__main__":
    import uvicorn
    # Make sure the 'app' below matches the name of your FastAPI() object
    uvicorn.run(app, host="0.0.0.0", port=7860)

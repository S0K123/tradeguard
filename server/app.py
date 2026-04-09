# server/app.py
from fastapi import FastAPI
import asyncio
from inference import main as run_inference

app = FastAPI()

@app.get("/")
def home():
    return {"status": "TradeGuard is running"}

@app.post("/reset")
async def reset():
    try:
        run_inference()   
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def main():
    import uvicorn
    # 'app' must be the name of your FastAPI() variable
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()

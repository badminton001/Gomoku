from fastapi import FastAPI

app = FastAPI(title="Gomoku-AI")

@app.get("/")
def root():
    return {"message": "Gomoku AI Backend"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI

app = FastAPI(title="multimodal-dialect-emotion")


@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi import FastAPI
from pydantic import BaseModel

class MusicRequest(BaseModel):
    prompt: str
    seconds: int

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/gio")
def handle_gio(mr: MusicRequest):
    # Echo the received request for easy testing
    return {"prompt": mr.prompt, "seconds": mr.seconds}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0")
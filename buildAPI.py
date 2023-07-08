import uvicorn
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.get("/")
def index():
    return {"ok": "Parece que anda bien!"}

@app.post("/file/upload")
def upload_file(file: UploadFile):
    data = file.file
    return {"content": data, "filename": file.filename}

#falta agregar que si no es la extension correcta que mande un mensaje de "error! incorrect file type"

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
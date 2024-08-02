from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from query_handler import question_with_chat_state
import os

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    chat_state: list
    table_name: str

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    try:
        response_generator = question_with_chat_state(
            question=request.question,
            chat_state=request.chat_state,
            table_name=request.table_name
        )
        return StreamingResponse(response_generator, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
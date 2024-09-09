import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from query_handler import question_with_chat_state, retrieve_similar_chunks

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    chat_state: list
    similar_chunks: list[dict]

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    try:
        response_generator = question_with_chat_state(
            question=request.question,
            chat_state=request.chat_state,
            similar_chunks=request.similar_chunks
        )
        return StreamingResponse(response_generator, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChunksRequest(BaseModel):
    question: str
    chat_state: list = []
    table_name: str

@app.post("/retrieve_chunks")
async def retrieve_chunks_endpoint(
    request: ChunksRequest
) -> dict:
    
    """
    Retrieves chunks similar to the given question.

    Args:
        request (ChunksRequest): Request with the question and the chat state.

    Returns:
        dict: A dictionary with a key "chunks" containing the similar chunks.

    Raises:
        HTTPException: If there is an error while retrieving the chunks, it raises
            an HTTPException with status code 500 and the error message as detail.
    """
    try:
        similar_chunks = retrieve_similar_chunks(
            table_name=request.table_name,
            question=request.question,
            chat_messages=request.chat_state
        )
        return {
            "chunks": similar_chunks
        }
    except Exception as e:
        logging.error(f"Error in retrieve_chunks_endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=str(e)
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
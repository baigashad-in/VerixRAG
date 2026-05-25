# streaming.py
"""
Server-Sent Events (SSE) streaming for token-by-token output.

WHY SSE OVER WEBSOCKETS?
- SSE is simpler (one-direction: server → client)
- Works over standard HTTP — no upgrade handshake
- Built-in browser support via EventSource API
- Perfect for LLM streaming where only the server sends data

WebSockets are better when the client also needs to send data 
during the stream (e.g., cancellation). For basic chat, SSE wins.
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from litellm import completion
import json
import asyncio

app = FastAPI()


async def stream_rag_response(query: str, context: str, 
                               system_prompt: str, model: str):
    """Yield tokens as Server-Sent Events.
    
    SSE FORMAT:
    Each event is a line starting with "data: " followed by JSON.
    A blank line separates events. The client reads these as 
    they arrive, building the response incrementally.
    
    data: {"token": "The", "type": "text"}
    
    data: {"token": " refund", "type": "text"}
    
    data: {"type": "done", "citations": [...]}
    """
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Sources:\n{context}
            \n\nQuestion: {query}\n\nAnswer with citations."""},
        ],
        temperature=0.0,
        stream=True,  # this is the key flag
    )
    
    full_response = ""
    
    for chunk in response:
        # Each chunk contains a small piece of the response
        delta = chunk.choices[0].delta
        
        if hasattr(delta, "content") and delta.content:
            token = delta.content
            full_response += token
            
            yield f"data: {json.dumps({'token': token, 'type': 'text'})}\n\n"
    
    # Send completion signal with metadata
    yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Main chat endpoint with streaming.
    
    The frontend calls this with a query. The response streams 
    back token-by-token. The frontend appends each token to 
    build the response in real-time.
    """
    body = await request.json()
    query = body["query"]
    
    # --- In production, wire up the full pipeline here ---
    # results = enhanced_retriever.retrieve(query, ...)
    # context, citations = generator._build_context_block(results)
    # For now, showing the streaming mechanism:
    
    context = "placeholder — wire up retrieval pipeline"
    system_prompt = RAGGenerator()._build_system_prompt()
    model = "anthropic/claude-sonnet-4-20250514"
    
    return StreamingResponse(
        stream_rag_response(query, context, system_prompt, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
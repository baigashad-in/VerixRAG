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

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from litellm import completion
from collections import defaultdict
from src.generation.generator import RAGGenerator
import json
import asyncio
import time

app = FastAPI()

# In memory rate limiter
_rate_limits = dict[str, list[float]] = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 20


def _check_rate_limit(client_ip: str):
    """Prevent abuse - limit reuests per IP."""
    now = time.time()
    window = [t for t in _rate_limits[client_ip] if now - t < 60]
    _rate_limits[client_ip] = window

    if len(window) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again in a minute."
        )
    _rate_limits[client_ip].append(now)

def _escape_sse_data(data: dict) -> str:
    """Escape data for safe SSE to prevent injection.
    
    SSE uses newlines as delimiters. If the LLM generates
    a toke containing 'data:', it could inject fake events.
    json.dumps handles this beacuse it escapes newlines.
    """
    return f"data: {json.dumps(data, ensure_ascii=True)}\n\n"


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
            {"role": "user", "content": f"""<SOURCES>:\n{context}</SOURCES>
            \n\n <USER_QUERY> Question: {query} </USER_QUERY>\n\nAnswer with citations."""},
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
            
            yield _escape_sse_data({'token': token, 'type': 'text'})
    
    # Send completion signal with metadata
    yield _escape_sse_data({'type': 'done'})


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Main chat endpoint with streaming.
    
    The frontend calls this with a query. The response streams 
    back token-by-token. The frontend appends each token to 
    build the response in real-time.
    """
    _check_rate_limit(request.client.host)

    body = await request.json()
    query = body["query"]
    
    if not query:
        raise HTTPException(400, "Query is required.")
    if len(query) > 2000:
        raise HTTPException(400, "Query too long (max 2000 chars).")
    
    # --- In production, wire up the full pipeline here ---
    # results = enhanced_retriever.retrieve(query, ...)
    # context, citations = generator._build_context_block(results)
    # For now, showing the streaming mechanism:
    
    context = "placeholder — wire up retrieval pipeline"
    system_prompt = RAGGenerator()._build_system_prompt()
    model = "groq/llama-3.3-70b-versatile"
    
    return StreamingResponse(
        stream_rag_response(query, context, system_prompt, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff",
        },
    )
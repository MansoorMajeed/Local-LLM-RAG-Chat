#!/usr/bin/env python3
import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-http-server")

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434") 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# Initialize FastAPI app
app = FastAPI(title="Local RAG API", description="HTTP API for RAG document search")

# Global clients
chroma_client = None
collection = None

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    min_similarity: float = 0.3

class SearchResponse(BaseModel):
    results: str

class DocumentInfoRequest(BaseModel):
    file_name: Optional[str] = None

class DocumentInfoResponse(BaseModel):
    info: str

async def init_clients():
    """Initialize Chroma client and collection"""
    global chroma_client, collection
    if not chroma_client:
        try:
            chroma_client = chromadb.HttpClient(
                host=CHROMA_HOST.split(':')[0],
                port=int(CHROMA_HOST.split(':')[1])
            )
            collection = chroma_client.get_collection("documents")
            logger.info("Connected to ChromaDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

async def get_embeddings(text: str) -> List[float]:
    """Get embeddings for text using Ollama"""
    try:
        # Create Ollama client with proper host
        client = ollama.Client(host=f"http://{OLLAMA_HOST}")
        response = client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    await init_clients()

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search through documents using vector similarity"""
    try:
        await init_clients()
        
        # Get query embedding
        query_embedding = await get_embeddings(request.query)
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.limit,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return SearchResponse(results="No documents found matching your query.")
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # Convert cosine distance to similarity
            similarity = max(0, 1 - distance)
            if similarity >= request.min_similarity:
                formatted_results.append(
                    f"**Result {i+1}** (similarity: {similarity:.3f})\n"
                    f"**Source:** {metadata.get('file_name', 'Unknown')}\n"
                    f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\n"
                )
        
        if not formatted_results:
            return SearchResponse(results=f"No documents found with similarity >= {request.min_similarity}")
        
        result_text = f"Found {len(formatted_results)} relevant documents:\n\n" + "\n---\n".join(formatted_results)
        return SearchResponse(results=result_text)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {e}")

@app.post("/document-info", response_model=DocumentInfoResponse)
async def get_document_info(request: DocumentInfoRequest):
    """Get information about indexed documents"""
    try:
        await init_clients()
        
        if request.file_name:
            # Get info for specific file
            results = collection.get(
                where={"file_name": request.file_name},
                include=["metadatas"]
            )
            if not results['ids']:
                return DocumentInfoResponse(info=f"No document found with name: {request.file_name}")
            
            info_text = (
                f"Document: {request.file_name}\n"
                f"Chunks: {len(results['ids'])}\n"
                f"Last processed: {results['metadatas'][0].get('processed_at', 'Unknown')}"
            )
            return DocumentInfoResponse(info=info_text)
        else:
            # Get overall collection info
            count = collection.count()
            
            # Get unique files
            all_metadata = collection.get(include=["metadatas"])
            unique_files = set()
            for metadata in all_metadata['metadatas']:
                unique_files.add(metadata.get('file_name', 'Unknown'))
            
            info_text = (
                f"RAG Database Status:\n"
                f"Total chunks: {count}\n"
                f"Unique documents: {len(unique_files)}\n"
                f"Documents: {', '.join(sorted(unique_files))}"
            )
            return DocumentInfoResponse(info=info_text)
            
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document info: {e}")

@app.get("/")
async def root():
    return {"message": "Local RAG HTTP API", "endpoints": ["/search", "/document-info", "/docs", "/openapi.json"]}

@app.get("/openapi.json")
async def get_openapi_spec():
    """Return OpenAPI specification for tool integration"""
    import json
    try:
        with open('/app/openapi.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback: generate spec dynamically
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Local RAG API",
                "description": "Search through personal documents using vector similarity",
                "version": "1.0.0"
            },
            "servers": [{"url": "http://localhost:8004"}],
            "paths": {
                "/search": {
                    "post": {
                        "summary": "Search Documents",
                        "description": "Search through documents using vector similarity",
                        "operationId": "searchDocuments",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["query"],
                                        "properties": {
                                            "query": {"type": "string", "description": "Search query"},
                                            "limit": {"type": "integer", "default": 5},
                                            "min_similarity": {"type": "number", "default": 0.3}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Search results",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "results": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use different port for HTTP API
    port = int(os.getenv("HTTP_PORT", "8003"))
    
    logger.info("Starting Local RAG HTTP Server")
    logger.info(f"Configuration:")
    logger.info(f"  CHROMA_HOST: {CHROMA_HOST}")
    logger.info(f"  OLLAMA_HOST: {OLLAMA_HOST}")
    logger.info(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    logger.info(f"  HTTP_PORT: {port}")
    
    uvicorn.run(app, host='0.0.0.0', port=port)
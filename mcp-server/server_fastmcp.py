#!/usr/bin/env python3
import os
import logging
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP
import chromadb
import ollama

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-mcp-server")

# Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost:8000")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434") 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# Initialize FastMCP server
mcp = FastMCP("local-rag")

# Global clients
chroma_client = None
collection = None

async def init_clients():
    """Initialize Chroma client and collection"""
    global chroma_client, collection
    if not chroma_client:
        try:
            chroma_client = chromadb.HttpClient(
                host=CHROMA_HOST.split(':')[0], 
                port=int(CHROMA_HOST.split(':')[1])
            )
            collection = chroma_client.get_or_create_collection(
                "documents",
                metadata={"description": "Local RAG document collection", "hnsw:space": "cosine"}
            )
            logger.info("Initialized Chroma client and collection")
        except Exception as e:
            logger.error(f"Error connecting to Chroma: {e}")
            raise

async def get_embeddings(text: str) -> List[float]:
    """Get embeddings from Ollama"""
    try:
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise

@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.001
) -> str:
    """Search through indexed documents using semantic similarity
    
    Args:
        query: The search query
        limit: Maximum number of results to return (default: 5)
        min_similarity: Minimum similarity score from 0.0 to 1.0 (default: 0.001)
    
    Returns:
        Formatted search results with similarity scores
    """
    await init_clients()
    
    try:
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Search in Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return "No documents found matching your query."
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # Convert cosine distance to similarity
            similarity = max(0, 1 - distance)
            if similarity >= min_similarity:
                formatted_results.append(
                    f"**Result {i+1}** (similarity: {similarity:.3f})\n"
                    f"**Source:** {metadata.get('file_name', 'Unknown')}\n"
                    f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\n"
                )
        
        if not formatted_results:
            return f"No documents found with similarity >= {min_similarity}"
        
        return f"Found {len(formatted_results)} relevant documents:\n\n" + "\n---\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Error searching documents: {e}"

@mcp.tool()
async def get_document_info(
    file_name: str = None
) -> str:
    """Get information about indexed documents
    
    Args:
        file_name: Optional specific file name to get info for
    
    Returns:
        Document information or overall collection statistics
    """
    await init_clients()
    
    try:
        if file_name:
            # Get info for specific file
            results = collection.get(
                where={"file_name": file_name},
                include=["metadatas"]
            )
            if not results['ids']:
                return f"No document found with name: {file_name}"
            
            return (
                f"Document: {file_name}\n"
                f"Chunks: {len(results['ids'])}\n"
                f"Last processed: {results['metadatas'][0].get('processed_at', 'Unknown')}"
            )
        else:
            # Get overall collection info
            count = collection.count()
            
            # Get unique files
            all_metadata = collection.get(include=["metadatas"])
            unique_files = set()
            for metadata in all_metadata['metadatas']:
                unique_files.add(metadata.get('file_name', 'Unknown'))
            
            return (
                f"RAG Database Status:\n"
                f"Total chunks: {count}\n"
                f"Unique documents: {len(unique_files)}\n"
                f"Documents: {', '.join(sorted(unique_files))}"
            )
            
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        return f"Error getting document info: {e}"

@mcp.tool()
async def search_by_source(
    query: str,
    source_filter: str,
    limit: int = 5
) -> str:
    """Search within specific document sources
    
    Args:
        query: The search query
        source_filter: Filter by source file name or path
        limit: Maximum number of results (default: 5)
    
    Returns:
        Formatted search results from the filtered source
    """
    await init_clients()
    
    try:
        # Get query embedding
        query_embedding = await get_embeddings(query)
        
        # Search with source filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit * 2,  # Get more to filter
            where={"file_name": {"$contains": source_filter}},
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return f"No documents found in source containing '{source_filter}'"
        
        # Format results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0][:limit],
            results['metadatas'][0][:limit],
            results['distances'][0][:limit]
        )):
            # Convert cosine distance to similarity
            similarity = max(0, 1 - distance)
            formatted_results.append(
                f"**Result {i+1}** (similarity: {similarity:.3f})\n"
                f"**Source:** {metadata.get('file_name', 'Unknown')}\n" 
                f"**Content:** {doc[:500]}{'...' if len(doc) > 500 else ''}\n"
            )
        
        return (
            f"Found {len(formatted_results)} results in '{source_filter}':\n\n" +
            "\n---\n".join(formatted_results)
        )
        
    except Exception as e:
        logger.error(f"Error searching by source: {e}")
        return f"Error searching by source: {e}"

if __name__ == "__main__":
    import sys
    
    logger.info("Starting Local RAG MCP Server with FastMCP")
    logger.info(f"Configuration:")
    logger.info(f"  CHROMA_HOST: {CHROMA_HOST}")
    logger.info(f"  OLLAMA_HOST: {OLLAMA_HOST}")
    logger.info(f"  EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    
    # Check if HTTP mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        logger.info("Starting HTTP server on port 8002")
        # Use FastAPI directly since FastMCP might not support HTTP transport
        import uvicorn
        app = mcp.create_app()
        uvicorn.run(app, host='0.0.0.0', port=8002)
    else:
        logger.info("Starting stdio transport for MCP")
        # Run the server using stdio transport
        mcp.run(transport='stdio')
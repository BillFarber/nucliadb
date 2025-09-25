#!/usr/bin/env python3
"""
Simple gRPC client to delete a knowledge box using DeleteKnowledgeBox service.
"""

import asyncio

import grpc
from grpc import aio

from nucliadb_protos import knowledgebox_pb2, writer_pb2_grpc


async def delete_knowledge_box():
    """Delete knowledge box using DeleteKnowledgeBox service."""
    print("🗑️  Deleting Knowledge Box...")
    
    # Connect to gRPC service
    channel = aio.insecure_channel("localhost:8030")
    
    try:
        writer_stub = writer_pb2_grpc.WriterStub(channel)
        
        # Create the knowledge box ID request
        kb_id = knowledgebox_pb2.KnowledgeBoxID(
            slug="new_knowledge_box_v2",
            uuid="12341234-1234-1234-1234-123412341234"
        )
        
        print(f"📡 Deleting KB - Slug: {kb_id.slug}")
        print(f"📡 Deleting KB - UUID: {kb_id.uuid}")
        
        # Send the delete request
        response = await writer_stub.DeleteKnowledgeBox(kb_id)
        
        print(f"✅ Delete Response Status: {response.status}")
        if hasattr(response, 'error_message') and response.error_message:
            print(f"💬 Message: {response.error_message}")
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(delete_knowledge_box())
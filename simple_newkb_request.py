#!/usr/bin/env python3
"""
Simple gRPC client to send NewKnowledgeBoxV2 request with specific data.
"""

import asyncio

import grpc
from grpc import aio

from nucliadb_protos import utils_pb2, writer_pb2, writer_pb2_grpc


async def send_new_kb_request():
    """Send NewKnowledgeBoxV2 request with the specified data."""
    print("🚀 Sending NewKnowledgeBoxV2 request...")
    
    # Connect to gRPC service
    channel = aio.insecure_channel("localhost:8030")
    
    try:
        writer_stub = writer_pb2_grpc.WriterStub(channel)
        
        # Create vector set with DOT similarity
        vector_set = writer_pb2.NewKnowledgeBoxV2Request.VectorSet(
            vectorset_id="multilingual-2024-05-06",
            similarity=utils_pb2.VectorSimilarity.DOT,
            vector_dimension=512,
            matryoshka_dimensions=[768, 1024]
        )
        
        # Create the request
        request = writer_pb2.NewKnowledgeBoxV2Request(
            kbid="12341234-1234-1234-1234-123412341234",
            slug="new_knowledge_box_v2",
            title="new_knowledge_box_v2",
            description="new_knowledge_box_v2",
            vectorsets=[vector_set],
            hidden_resources_enabled=True,
            hidden_resources_hide_on_creation=True
        )
        
        print(f"📡 KB ID: {request.kbid}")
        print(f"📡 Title: {request.title}")
        print(f"📡 Vector Set: {request.vectorsets[0].vectorset_id}")
        print(f"📡 Similarity: {request.vectorsets[0].similarity}")
        
        # Send the request
        response = await writer_stub.NewKnowledgeBoxV2(request)
        
        print(f"✅ Response Status: {response.status}")
        if response.error_message:
            print(f"💬 Message: {response.error_message}")
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(send_new_kb_request())
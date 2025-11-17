#!/usr/bin/env python3
"""
Simple test client for LLM WebSocket service.
Connects to the LLM service and generates tokens from a prompt.
"""

import asyncio
import websockets
import json
import sys


async def test_llm_generation():
    """Test LLM token generation via WebSocket."""
    
    # Connect to the WebSocket server
    uri = "ws://localhost:8000/ws/llm"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("✓ Connected to LLM service")
        
        # Receive session_started message
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ {data.get('type')}: Session ID = {data.get('session_id')}")
        session_id = data.get('session_id')
        
        # Start session with optional system prompt
        start_message = {
            "type": "start_session",
            "config": {
                "system_prompt": "You are a helpful AI assistant. Be concise and friendly."
            }
        }
        await websocket.send(json.dumps(start_message))
        print("✓ Sent session configuration")
        
        # Receive session_configured response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✓ {data.get('type')}")
        
        # Send user message
        user_prompt = input("\nEnter your prompt (or press Enter for default): ").strip()
        if not user_prompt:
            user_prompt = "Write a haiku about coding."
        
        print("\n🤖 Generating response for: '{user_prompt}'\n")
        
        user_message = {
            "type": "user_message",
            "text": user_prompt
        }
        await websocket.send(json.dumps(user_message))
        
        # Receive streaming tokens
        full_response = ""
        token_count = 0
        
        print("Response: ", end="", flush=True)
        import sys
        
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            msg_type = data.get("type")
            
            if msg_type == "token":
                # Stream token to stdout - server sends tokens in "data" field
                token = data.get("data", "")
                # Print with visual indicator to show streaming
                print(f"\033[92m{token}\033[0m", end="", flush=True)
                sys.stdout.flush()  # Force flush to show immediately
                full_response += token
                token_count += 1
                # Add visible delay to see streaming effect
                await asyncio.sleep(0.08)  # 80ms delay makes streaming visible
                
            elif msg_type == "response_complete":  # Changed from generation_complete
                print("\n")
                stats = data.get("stats", {})
                print(f"\n✓ Generation complete!")
                print(f"  - Total tokens: {stats.get('tokens_generated', token_count)}")
                print(f"  - Generation time: {stats.get('processing_time_ms', 0)/1000:.2f}s")
                if stats.get('tokens_per_second'):
                    print(f"  - Tokens/second: {stats.get('tokens_per_second', 0):.1f}")
                break
                
            elif msg_type == "error":
                error = data.get("error", {})
                print(f"\n✗ Error: {error.get('message')}")
                print(f"  Code: {error.get('code')}")
                break
            
            else:
                print(f"\nReceived: {msg_type}")
        
        # End session
        end_message = {"type": "end_session"}
        await websocket.send(json.dumps(end_message))
        print("\n✓ Session ended")


async def check_health():
    """Check if the LLM service is healthy."""
    import aiohttp
    
    health_url = "http://localhost:8000/health"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✓ LLM Service is healthy")
                    print(f"  Status: {data.get('status')}")
                    return True
                else:
                    print(f"✗ Health check failed: {response.status}")
                    return False
    except Exception as e:
        print(f"✗ Cannot connect to LLM service: {e}")
        print(f"  Make sure the service is running on http://localhost:8000")
        return False


async def main():
    """Main entry point."""
    print("=" * 60)
    print("FastTalk LLM Service Test Client")
    print("=" * 60)
    
    # Check health first
    print("\nChecking service health...")
    healthy = await check_health()
    
    if not healthy:
        print("\n⚠️  Service not available. Please start the LLM service first.")
        sys.exit(1)
    
    # Run test
    try:
        await test_llm_generation()
    except websockets.exceptions.WebSocketException as e:
        print(f"\n✗ WebSocket error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n✓ Test interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check if websockets and aiohttp are available
    try:
        import websockets
        import aiohttp
    except ImportError:
        print("Installing required dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "aiohttp"])
        print("✓ Dependencies installed. Please run the script again.")
        sys.exit(0)
    
    asyncio.run(main())

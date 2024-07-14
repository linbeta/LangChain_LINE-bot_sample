from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag import rag_chain
from pydantic import BaseModel
import asyncio
from linebot.v3 import (
    WebhookParser
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.messaging import (
    Configuration,
    AsyncApiClient,
    AsyncMessagingApi,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent
)
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

# Configurations in Dev
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chain, path="/rag")


Configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
api_client = AsyncApiClient(Configuration)
line_bot_api = AsyncMessagingApi(api_client)
parser = WebhookParser(LINE_CHANNEL_SECRET)


class CallbackRequest(BaseModel):
    events: list
    destination: str

class RagInvokeRequest(BaseModel):
    input: dict


@app.post("/callback")
async def callback(request: Request):

    x_line_signature = request.headers['X-Line-Signature']

    body = await request.body()
    body = body.decode('utf-8')
    # print("Body: ", body)
    try:
        events = parser.parse(body, x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessageContent):
            continue

        async def call_rag_invoke(input_text):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/rag/invoke", 
                    json={"input": {"input": input_text}}, 
                    headers={"Content-Type": "application/json"},
                    timeout=10.0
                )
                response_json = response.json()
                # print("[DEBUG]>>>Response JSON: ", response_json)
                return response_json
        
        response = await call_rag_invoke(event.message.text)

        reply_text = response['output']['answer']
        # print("Reply Text: ", reply_text)
        await line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )

    return "OK"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

from __future__ import annotations

import json
import os
import httpx

from boardforge.agent import JobAgent
from boardforge.models import FeedbackRequest


TOOLS = [
    {
        "type":"function",
        "function":{
            "name":"search_jobs",
            "description":"Search and rank currently active jobs using the candidate's saved preferences.",
            "parameters":{
                "type":"object",
                "properties":{"query":{"type":"string"},"top_k":{"type":"integer","minimum":1,"maximum":20}},
                "required":["query"],
                "additionalProperties":False,
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"update_preferences",
            "description":"Persist a durable candidate preference or constraint learned from conversation.",
            "parameters":{
                "type":"object",
                "properties":{"text":{"type":"string"}},
                "required":["text"],
                "additionalProperties":False,
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"record_feedback",
            "description":"Record feedback on a specific stable job id.",
            "parameters":{
                "type":"object",
                "properties":{
                    "job_id":{"type":"string"},
                    "verdict":{"type":"string","enum":["save","reject","applied","interview","ignore"]},
                    "note":{"type":"string"},
                },
                "required":["job_id","verdict"],
                "additionalProperties":False,
            },
        },
    },
]


class ToolCallingAgent:
    def __init__(self, core: JobAgent):
        self.core = core
        self.key = os.getenv("OPENAI_API_KEY")
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    async def run(self, profile_id: str, text: str) -> dict:
        if not self.key:
            result = await self.core.handle(profile_id, text)
            return {"mode":"deterministic","message":result.message,**result.data}

        messages = [
            {
                "role":"system",
                "content":(
                    "You are a job-search agent. Use tools instead of inventing job facts. "
                    "Persist durable preferences when the user changes them. Never claim a job attribute "
                    "that is not returned by a tool. Be concise and evidence-based."
                ),
            },
            {"role":"user","content":text},
        ]

        for _ in range(5):
            payload = {
                "model":self.model,
                "messages":messages,
                "tools":TOOLS,
                "tool_choice":"auto",
            }
            async with httpx.AsyncClient(timeout=45) as client:
                r = await client.post(
                    f"{self.base}/chat/completions",
                    headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"},
                    json=payload,
                )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                return {"mode":"llm-tools","message":msg.get("content","")}

            messages.append(msg)
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                if name == "search_jobs":
                    result = await self.core.search_jobs(profile_id, args.get("query",""), args.get("top_k",10))
                    output = {"message":result.message,**result.data}
                elif name == "update_preferences":
                    result = await self.core.update_preferences(profile_id, args["text"])
                    output = {"message":result.message,"preferences":result.data}
                elif name == "record_feedback":
                    result = await self.core.record_feedback(FeedbackRequest(
                        profile_id=profile_id,
                        job_id=args["job_id"],
                        verdict=args["verdict"],
                        note=args.get("note",""),
                    ))
                    output = {"message":result.message,**result.data}
                else:
                    output = {"error":f"unknown tool {name}"}

                messages.append({
                    "role":"tool",
                    "tool_call_id":call["id"],
                    "content":json.dumps(output, default=str),
                })

        return {"mode":"llm-tools","message":"Tool-call limit reached."}

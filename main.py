from fastapi import FastAPI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import streamlit as st
import requests
import numpy as np
import threading
import uvicorn

app = FastAPI()

# Milvus 클라이언트 설정
connections.connect("default", host="localhost", port="19530")
collection_name = "bug_reports"

# 컬렉션 스키마 설정
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=255)
]
# 컬렉션 생성
schema = CollectionSchema(fields, "Bug reports collection")
collection = Collection(collection_name, schema)


@app.on_event("startup")
def startup_event():
    if not collection.is_empty:
        collection.drop()
    collection.create_index("embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
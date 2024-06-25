from fastapi import FastAPI, HTTPException, Query
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import logging

app = FastAPI()

# Milvus 클라이언트 설정
connections.connect("default", host="localhost", port="19530")
collection_name = "bug_reports"

# 컬렉션이 이미 존재하는지 확인하고 존재하지 않으면 생성
if not utility.has_collection(collection_name):
    # 컬렉션 스키마 설정
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048)
    ]
    schema = CollectionSchema(fields, "Bug reports collection")

    # 새로운 컬렉션 생성
    collection = Collection(collection_name, schema)
else:
    collection = Collection(collection_name)

# Sentence Transformer 모델 로드
model_name = "jhgan/ko-sroberta-multitask"
sentence_transformer_ef = SentenceTransformer(model_name)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_event():
    try:
        if collection.is_empty:
            logger.info("Creating index for collection")
            collection.create_index("embedding", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
        collection.load()
    except Exception as e:
        logger.error(f"Error during startup: {e}")


class BugReport(BaseModel):
    작성자: str
    고객사: str
    업무: List[str]
    버그_내용: str
    날짜: str


class SearchQuery(BaseModel):
    query_text: str


@app.post("/add_bug/")
def add_bug(report: BugReport):
    try:
        metadata = f"작성자: {report.작성자} | 고객사: {report.고객사} | 업무: {', '.join(report.업무)} | 날짜: {report.날짜} | 버그 내용: {report.버그_내용}"
        embedding = sentence_transformer_ef.encode([report.버그_내용])[0].tolist()
        collection.insert([[embedding], [metadata]])
        collection.load()
        return {"message": "Bug report added successfully"}
    except Exception as e:
        logger.error(f"Error adding bug report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_bug/")
def search_bug(query: SearchQuery):
    try:
        embedding = sentence_transformer_ef.encode([query.query_text])[0].tolist()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        results = collection.search([embedding], "embedding", search_params, limit=3)
        response = []
        for result in results:
            for res in result:
                metadata = res.entity.get("metadata")
                parsed_metadata = parse_metadata(metadata)
                response.append({
                    "작성자": parsed_metadata.get("작성자"),
                    "고객사": parsed_metadata.get("고객사"),
                    "업무": parsed_metadata.get("업무"),
                    "버그_내용": parsed_metadata.get("버그 내용"),
                    "거리": res.distance
                })
        return response
    except Exception as e:
        logger.error(f"Error searching bug report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_metadata(metadata):
    """메타데이터 필드를 파싱하여 딕셔너리로 반환"""
    fields = metadata.split(" | ")
    parsed_data = {}
    for field in fields:
        key, value = field.split(": ", 1)
        parsed_data[key] = value
    return parsed_data


@app.get("/bug_reports")
def bug_reports():
    try:
        # 모든 엔트리 가져오기
        bug_reports = collection.query(expr="id >= 0", output_fields=["id", "metadata"])
        logger.info(f"Number of bug reports fetched: {len(bug_reports)}")

        return {
            "bug_reports": bug_reports
        }
    except Exception as e:
        logger.error(f"Error fetching bug reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_bug/{report_id}")
def delete_bug(report_id: int):
    try:
        result = collection.query(expr=f"id == {report_id}", output_fields=["id"])
        if not result:
            raise HTTPException(status_code=404, detail="Bug report not found")
        collection.delete(expr=f"id == {report_id}")
        return {"message": "Bug report deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting bug report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Bug Finder API"}

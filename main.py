from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from jsonschema import ValidationError
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from pydantic import BaseModel, ValidationError
from typing import List
from sentence_transformers import SentenceTransformer
import logging
import json

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
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2048),
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
            collection.create_index(
                "embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128},
                },
            )
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


def convert_numpy_float_to_python_float(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = convert_numpy_float_to_python_float(value)
    elif isinstance(data, list):
        data = [convert_numpy_float_to_python_float(item) for item in data]
    elif isinstance(data, np.float32):
        data = float(data)
    return data


@app.post("/add_bug/")
def add_bug(report: BugReport):
    try:
        # 메타데이터를 JSON 형태로 변환
        metadata = {
            "작성자": report.작성자,
            "고객사": report.고객사,
            "업무": report.업무,
            "날짜": report.날짜,
            "버그_내용": report.버그_내용,
        }
        # JSON 객체를 문자열로 변환하여 저장
        metadata_str = json.dumps(metadata, ensure_ascii=False)
        embedding = sentence_transformer_ef.encode([report.버그_내용])[0].tolist()
        collection.insert([[embedding], [metadata_str]])
        collection.flush()
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
            "params": {"nprobe": 10},
        }
        results = collection.search(
            [embedding], "embedding", search_params, limit=3, output_fields=["metadata"]
        )

        response = []
        for result in results:
            for res in result:
                logger.info(f"Result object: {res}")
                metadata_str = res.entity.get("metadata")
                if metadata_str is not None:
                    metadata = json.loads(metadata_str)
                    response.append(
                        {
                            "작성자": metadata.get("작성자"),
                            "고객사": metadata.get("고객사"),
                            "업무": metadata.get("업무"),
                            "버그_내용": metadata.get("버그_내용"),
                            "날짜": metadata.get("날짜"),
                            "거리": res.distance,
                        }
                    )
                else:
                    logger.error(f"Metadata is None for result ID: {res.id}")

        return response
    except Exception as e:
        logger.error(f"Error searching bug report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bug_reports/")
def bug_reports():
    try:
        bug_report_data = collection.query(
            expr="id > 0", output_fields=["id", "metadata"]
        )
        response = []
        for report in bug_report_data:
            metadata_str = report.get("metadata")
            metadata = json.loads(metadata_str)
            response.append(
                {
                    "id": report.get("id"),
                    "작성자": metadata.get("작성자"),
                    "고객사": metadata.get("고객사"),
                    "업무": metadata.get("업무"),
                    "버그_내용": metadata.get("버그_내용"),
                    "날짜": metadata.get("날짜"),
                }
            )

        logger.info(f"Number of bug reports fetched: {len(response)}")
        return {"bug_reports": response}
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
        collection.flush()
        collection.load()
        return {"message": "Bug report deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting bug report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add_bug_reports_from_json/")
async def add_bug_reports(reports: List[BugReport]):
    try:

        # 유효한 리포트를 Milvus에 추가
        for report in reports:
            metadata = {
                "작성자": report.작성자,
                "고객사": report.고객사,
                "업무": report.업무,
                "날짜": report.날짜,
                "버그_내용": report.버그_내용,
            }
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            embedding = sentence_transformer_ef.encode([report.버그_내용])[0].tolist()
            collection.insert([[embedding], [metadata_str]])

        collection.flush()
        collection.load()
        return {
            "message": "Bug reports added successfully",
            "count": len(reports),
        }
    except ValidationError as e:
        logger.error(f"Invalid report format: {e.json()}")
        raise HTTPException(status_code=400, detail="Invalid report format")
    except Exception as e:
        logger.error(f"Error adding bug reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 전체 컬렉션 삭제
@app.delete("/delete_all/")
def delete_all():
    try:
        collection.drop()
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_bug/{report_id}")
def get_bug(report_id: int):
    try:
        # 특정 id에 해당하는 문서 데이터를 조회
        result = collection.query(expr=f"id == {report_id}", output_fields=["*"])
        if not result:
            raise HTTPException(status_code=404, detail="Bug report not found")
        # numpy.float32를 Python float로 변환
        result_converted = convert_numpy_float_to_python_float(result)
        # 조회된 데이터를 반환
        return jsonable_encoder({"data": result_converted})
    except Exception as e:
        logger.error(f"Error fetching bug report with id {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Bug Finder API"}

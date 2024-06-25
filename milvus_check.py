from pymilvus import connections, Collection

# Milvus 클라이언트 설정
connections.connect("default", host="localhost", port="19530")
collection_name = "bug_reports"

# 컬렉션 불러오기
collection = Collection(collection_name)

# 전체 데이터 조회
all_data = collection.query(expr="id >= 0", output_fields=["id", "metadata"])
print(f"Total data fetched: {len(all_data)}")
for data in all_data:
    print(data)

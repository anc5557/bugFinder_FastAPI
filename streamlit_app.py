import streamlit as st
import requests

# Streamlit 애플리케이션 제목 및 설명
st.title("버그 파인더")
st.write("버그를 검색하거나 새로운 버그 리포트를 추가하세요.")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["버그 검색", "버그 리포트 추가", "버그 리포트 목록"])


def parse_metadata(metadata):
    """메타데이터 필드를 파싱하여 딕셔너리로 반환"""
    fields = metadata.split(" | ")
    parsed_data = {}
    for field in fields:
        key, value = field.split(": ", 1)
        parsed_data[key] = value
    return parsed_data


def truncate_text(text, max_length=100):
    """텍스트를 지정된 길이로 줄이고 '...' 추가"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


with tab1:
    # 검색어 입력란
    query_text = st.text_input("검색어를 입력하세요")

    # 검색 버튼
    if st.button("검색"):
        if query_text:
            # FastAPI 서버에 검색 요청
            response = requests.post("http://localhost:8000/search_bug/", json={"query_text": query_text})

            if response.status_code == 200:
                results = response.json()
                st.write("검색 결과:")
                for result in results:
                    st.write(f"작성자: {result['작성자']}")
                    st.write(f"고객사: {result['고객사']}")
                    st.write(f"업무: {result['업무']}")
                    st.write(f"버그 내용: {result['버그_내용']}")
                    st.write(f"거리: {result['거리']}")
                    st.write("---")
            else:
                st.write("검색 중 오류가 발생했습니다.")
        else:
            st.write("검색어를 입력해주세요.")

with tab2:
    # 버그 리포트 추가 폼
    with st.form(key='bug_report_form'):
        작성자 = st.text_input("작성자")
        고객사 = st.text_input("고객사")
        업무 = st.text_input("업무 (쉼표로 구분하여 입력)")
        버그_내용 = st.text_area("버그 내용")
        날짜 = st.date_input("날짜")

        submit_button = st.form_submit_button(label='버그 리포트 추가')

        if submit_button:
            if 작성자 and 고객사 and 업무 and 버그_내용 and 날짜:
                # 입력된 데이터를 FastAPI 서버로 전송
                업무_list = [x.strip() for x in 업무.split(",")]
                report_data = {
                    "작성자": 작성자,
                    "고객사": 고객사,
                    "업무": 업무_list,
                    "버그_내용": 버그_내용,
                    "날짜": str(날짜)
                }
                response = requests.post("http://localhost:8000/add_bug/", json=report_data)
                if response.status_code == 200:
                    st.success("버그 리포트가 성공적으로 추가되었습니다.")
                else:
                    st.error("버그 리포트 추가 중 오류가 발생했습니다.")
            else:
                st.error("모든 필드를 입력해주세요.")

with tab3:
    # FastAPI 서버에서 버그 리포트 가져오기
    response = requests.get(f"http://localhost:8000/bug_reports")

    if response.status_code == 200:
        data = response.json()
        bug_reports = data["bug_reports"]

        # 자세히 보기 및 삭제 버튼 추가
        for report in bug_reports:
            parsed_metadata = parse_metadata(report["metadata"])
            truncated_bug_content = truncate_text(parsed_metadata.get("버그 내용", ""))

            expander_text = f"{parsed_metadata.get('고객사')} | {parsed_metadata.get('업무')} | {truncated_bug_content}"
            with st.expander(expander_text):

                st.write(f"작성자: {parsed_metadata.get('작성자')}")
                st.write(f"고객사: {parsed_metadata.get('고객사')}")
                st.write(f"업무: {parsed_metadata.get('업무')}")
                st.write(f"날짜: {parsed_metadata.get('날짜')}")
                st.write(f"버그 내용: {parsed_metadata.get('버그 내용')}")

                if st.button("삭제", key=f"delete_{report['id']}"):
                    delete_response = requests.delete(f"http://localhost:8000/delete_bug/{report['id']}")
                    if delete_response.status_code == 200:
                        st.success("버그 리포트가 성공적으로 삭제되었습니다.")
                    else:
                        st.error("버그 리포트 삭제 중 오류가 발생했습니다.")
    else:
        st.error("버그 리포트를 불러오는 중 오류가 발생했습니다.")

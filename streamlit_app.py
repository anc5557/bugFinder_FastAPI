import streamlit as st
import requests
import json

# Streamlit 애플리케이션 제목 및 설명
st.title("버그 파인더")
st.write("버그를 검색하거나 새로운 버그 리포트를 추가하세요.")

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(
    ["버그 검색", "버그 리포트 추가", "파일로 버그 리포트 추가", "버그 리포트 목록"]
)


def truncate_text(text, max_length=100):
    """텍스트를 지정된 길이로 줄이고 '...' 추가"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


with tab1:
    # 버그 검색 폼
    query_text = st.text_input("검색어를 입력하세요")

    if st.button("검색"):
        if query_text:
            response = requests.post(
                "http://localhost:8000/search_bug/", json={"query_text": query_text}
            )

            if response.status_code == 200:
                results = response.json()
                st.write("가장 비슷한 버그 레포트 3개입니다.")
                for id, result in enumerate(results, 1):
                    st.write(f"{id}:")
                    st.write(f"작성자: {result['작성자']}")
                    st.write(f"고객사: {result['고객사']}")
                    st.write(f"날짜: {result['날짜']}")
                    st.write(f"업무: {result['업무']}")
                    st.write(f"버그 내용: {result['버그_내용']}")
                    # st.write(f"거리: {result['거리']}")
                    st.write("---")
            else:
                st.write("검색 중 오류가 발생했습니다.")
        else:
            st.write("검색어를 입력해주세요.")

with tab2:
    # 버그 리포트 추가 폼
    with st.form(key="bug_report_form"):
        작성자 = st.text_input("작성자")
        고객사 = st.text_input("고객사")
        업무 = st.text_input("업무 (쉼표로 구분하여 입력)")
        버그_내용 = st.text_area("버그 내용")
        날짜 = st.date_input("날짜")

        submit_button = st.form_submit_button(label="버그 리포트 추가")

        if submit_button:
            if 작성자 and 고객사 and 업무 and 버그_내용 and 날짜:
                업무_list = [x.strip() for x in 업무.split(",")]
                report_data = {
                    "작성자": 작성자,
                    "고객사": 고객사,
                    "업무": 업무_list,
                    "버그_내용": 버그_내용,
                    "날짜": str(날짜),
                }
                response = requests.post(
                    "http://localhost:8000/add_bug/", json=report_data
                )
                if response.status_code == 200:
                    st.success("버그 리포트가 성공적으로 추가되었습니다.")
                else:
                    st.error("버그 리포트 추가 중 오류가 발생했습니다.")
            else:
                st.error("모든 필드를 입력해주세요.")

with tab3:
    # 파일로 버그 리포트 추가
    uploaded_file = st.file_uploader("버그 리포트 파일을 업로드하세요", type=["json"])
    st.write(
        "파일은 다음 필드를 포함해야 합니다: 작성자, 고객사, 업무, 버그_내용, 날짜"
    )
    st.write("날짜 형식: YYYY-MM-DD")
    st.write("업무는 리스트 형태로 입력해주세요")

    submit_button = st.button("버그 리포트 추가")

    if submit_button and uploaded_file is not None:
        try:
            reports = json.load(uploaded_file)
            if not isinstance(reports, list):
                st.error("JSON 데이터는 리포트 목록이어야 합니다.")
            else:
                response = requests.post(
                    "http://localhost:8000/add_bug_reports_from_json/",
                    json=reports,
                )
                if response.status_code == 200:
                    st.success("파일로부터 버그 리포트가 성공적으로 추가되었습니다.")
                    st.session_state.file_uploader = None
                else:
                    st.error(
                        f"파일로부터 버그 리포트 추가 중 오류가 발생했습니다: {response.json()}"
                    )
        except json.JSONDecodeError as e:
            st.error(f"유효하지 않은 JSON 형식입니다: {e}")
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

with tab4:
    # 버그 리포트 목록
    response = requests.get(f"http://localhost:8000/bug_reports")

    if response.status_code == 200:
        data = response.json()
        bug_reports = data["bug_reports"]

        if st.button("모든 버그 리포트 삭제"):
            delete_all_response = requests.delete("http://localhost:8000/delete_all/")
            if delete_all_response.status_code == 200:
                st.success("모든 버그 리포트가 성공적으로 삭제되었습니다.")
            else:
                st.error("버그 리포트 삭제 중 오류가 발생했습니다.")

        for report in bug_reports:
            truncated_bug_content = truncate_text(report.get("버그_내용", ""))

            tasks = ", ".join(report.get("업무", []))

            expander_text = (
                f"{report.get('고객사')} | {tasks} | {truncated_bug_content}"
            )
            with st.expander(expander_text):

                st.write(f"작성자: {report.get('작성자')}")
                st.write(f"고객사: {report.get('고객사')}")
                st.write(f"업무: {tasks}")
                st.write(f"날짜: {report.get('날짜')}")
                st.write(f"버그 내용: {report.get('버그_내용')}")

                if st.button("삭제", key=f"delete_{report['id']}"):
                    delete_response = requests.delete(
                        f"http://localhost:8000/delete_bug/{report['id']}"
                    )
                    if delete_response.status_code == 200:
                        st.success("버그 리포트가 성공적으로 삭제되었습니다.")
                    else:
                        st.error("버그 리포트 삭제 중 오류가 발생했습니다.")
    else:
        st.error("버그 리포트를 불러오는 중 오류가 발생했습니다.")

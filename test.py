import pandas as pd
import json
import os
import io
import re
from datetime import datetime

from sqlalchemy import create_engine, inspect
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


# =========================
# VISUALIZATION DETECTION
# =========================
def should_visualize(question):
    keywords = [
        "chart", "graph", "plot", "visualize", "trend", "compare",
        "distribution", "top", "bottom", "highest", "lowest",
        "month", "year", "sales", "revenue", "customer"
    ]
    return any(word in question.lower() for word in keywords)


# =========================
# SQL RESULT → DATAFRAME
# =========================
def sql_result_to_dataframe(results):
    if not results:
        return None

    labels, values = [], []

    for row in str(results).split("\n"):
        match = re.search(r"\('([^']+)',\s*([\d\.]+)\)", row)
        if match:
            labels.append(match.group(1))
            values.append(float(match.group(2)))

    if labels:
        return pd.DataFrame({"label": labels, "value": values})

    return None


# =========================
# CHART CREATION
# =========================
def create_chart(df, question):
    q = question.lower()

    if any(w in q for w in ["trend", "month", "year"]):
        return px.line(df, x="label", y="value", title="Trend Analysis")

    if any(w in q for w in ["distribution", "share"]):
        return px.pie(df, names="label", values="value", title="Distribution")

    if any(w in q for w in ["top", "highest", "lowest", "rank"]):
        return px.bar(df, x="value", y="label", orientation="h", title="Ranking")

    return px.bar(df, x="label", y="value", title="Comparison")


# =========================
# FILE HANDLING
# =========================
def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file), None
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file), None
        return None, "Unsupported file type"
    except Exception as e:
        return None, str(e)


def create_sql_database(data, table_name="business_data"):
    try:
        engine = create_engine("sqlite:///temp_business_data.db")
        data.to_sql(table_name, engine, if_exists="replace", index=False)

        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        schema = [(c["name"], str(c["type"])) for c in columns]

        return SQLDatabase(engine), schema, None
    except Exception as e:
        return None, None, str(e)


# =========================
# CHAT STORAGE
# =========================
def save_chat_to_file(chat_data, filename):
    os.makedirs("saved_chats", exist_ok=True)

    safe_msgs = []
    for m in chat_data["messages"]:
        clean = m.copy()
        clean.pop("chart", None)
        safe_msgs.append(clean)

    chat_data["messages"] = safe_msgs

    with open(os.path.join("saved_chats", filename), "w") as f:
        json.dump(chat_data, f, indent=2)

    return True


def load_chat_from_file(filename):
    path = os.path.join("saved_chats", filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def get_saved_chats():
    if not os.path.exists("saved_chats"):
        return []
    return sorted(
        [f for f in os.listdir("saved_chats") if f.endswith(".json")],
        reverse=True
    )


def delete_saved_chat(filename):
    path = os.path.join("saved_chats", filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# =========================
# AI ANALYSIS AGENT
# =========================
class AnalysisAgent:
    def __init__(self, database, api_key):
        self.database = database

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )

        sql_prompt = """You are an AI Data Analyst connected to a SQL database.

RULES:
- ONLY valid SQLite SELECT
- NO markdown
- NO explanations
- Use schema strictly
- Use aliases: label, value

Schema:
{schema}

Question:
{question}

SQL:
"""

        self.sql_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["schema", "question"],
                template=sql_prompt
            )
        )

        analysis_prompt = """You are a business analyst.

Data:
{summary}

Question:
{question}

Respond in:
1. What I Found
2. Why It Matters
3. Recommendations
"""

        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["question", "summary"],
                template=analysis_prompt
            )
        )

    def analyze(self, question):
        # Build schema text
        schema_text = ""
        for table in self.database.get_usable_table_names():
            schema_text += self.database.get_table_info([table])

        # Generate SQL safely
        sql_response = self.sql_chain.invoke({
            "schema": schema_text,
            "question": question
        })

        if isinstance(sql_response, dict):
            raw_sql = sql_response.get("text", "")
        else:
            raw_sql = str(sql_response)

        raw_sql = raw_sql.strip()

        match = re.search(r"(SELECT\s+.*)", raw_sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return {
                "status": "error",
                "analysis": f"SQL generation failed:\n{raw_sql}",
                "chart": None,
                "sql_results": None
            }

        sql_query = match.group(1).split(";")[0].strip()

        try:
            results = self.database.run(sql_query)
        except Exception as e:
            return {
                "status": "error",
                "analysis": f"SQL Execution Error: {e}",
                "chart": None,
                "sql_results": None
            }

        df = sql_result_to_dataframe(results)
        chart = create_chart(df, question) if df is not None and should_visualize(question) else None

        summary = df.to_string(index=False) if df is not None else str(results)

        analysis_response = self.analysis_chain.invoke({
            "question": question,
            "summary": summary
        })

        if isinstance(analysis_response, dict):
            analysis_text = analysis_response.get("text", "")
        else:
            analysis_text = str(analysis_response)

        analysis_text = analysis_text.strip()

        return {
            "status": "success",
            "sql_results": results,
            "analysis": analysis_text,
            "chart": chart
        }


# =========================
# PDF REPORT
# =========================
def generate_pdf_report(question, sql_results, analysis):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Business Data Analysis Report", styles["Heading1"]),
        Spacer(1, 12),
        Paragraph(f"<b>Question:</b> {question}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"<b>Results:</b><br/>{sql_results}", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(f"<b>Analysis:</b><br/>{analysis}", styles["BodyText"])
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer

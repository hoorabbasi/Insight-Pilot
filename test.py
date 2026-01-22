import pandas as pd
import json
import os
import io
import re
from datetime import datetime

from sqlalchemy import create_engine, inspect

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate

import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


# --------------------------------------------------
# VISUALIZATION DETECTION
# --------------------------------------------------
def should_visualize(question):
    keywords = [
        "chart", "graph", "plot", "visualize", "trend", "compare",
        "distribution", "top", "bottom", "highest", "lowest",
        "month", "year", "sales", "revenue", "customer"
    ]
    return any(word in question.lower() for word in keywords)


# --------------------------------------------------
# SQL RESULT → DATAFRAME
# --------------------------------------------------
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


# --------------------------------------------------
# CHART CREATION
# --------------------------------------------------
def create_chart(df, question):
    q = question.lower()

    if any(w in q for w in ["trend", "month", "year"]):
        return px.line(df, x="label", y="value", title="Trend Analysis")

    if any(w in q for w in ["distribution", "share"]):
        return px.pie(df, names="label", values="value", title="Distribution")

    if any(w in q for w in ["top", "highest", "lowest", "rank"]):
        return px.bar(df, x="value", y="label", orientation="h", title="Ranking")

    return px.bar(df, x="label", y="value", title="Comparison")


# --------------------------------------------------
# FILE READING
# --------------------------------------------------
def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file), None
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file), None
        return None, "Unsupported file type"
    except Exception as e:
        return None, str(e)


# --------------------------------------------------
# SQL DATABASE
# --------------------------------------------------
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


# --------------------------------------------------
# AI ANALYSIS AGENT
# --------------------------------------------------
class AnalysisAgent:
    def __init__(self, database, api_key):
        self.database = database

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )

        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
You are an AI Data Analyst connected to a SQL database.

RULES:
- Output ONE SQLite SELECT query
- NO markdown
- NO explanations

Schema:
{schema}

Question:
{question}

SQL:
"""
        )

        self.analysis_prompt = PromptTemplate(
            input_variables=["question", "summary"],
            template="""
You are a business analyst.

Data:
{summary}

Question:
{question}

Respond in 3 sections:
1. What I Found
2. Why It Matters
3. Recommendations
"""
        )

        self.sql_chain = self.sql_prompt | self.llm
        self.analysis_chain = self.analysis_prompt | self.llm

    # --------------------------------------------------
    # MAIN ANALYSIS
    # --------------------------------------------------
    def analyze(self, question):

        # -------- BUILD SCHEMA --------
        schema_text = ""
        for table in self.database.get_usable_table_names():
            schema_text += self.database.get_table_info([table])

        # -------- GENERATE SQL --------
        response = self.sql_chain.invoke({
            "schema": schema_text,
            "question": question
        })

        raw_text = response.content if hasattr(response, "content") else str(response)

        # -------- SAFE SQL EXTRACTION --------
        match = re.search(
            r"(SELECT\s+[\s\S]+?)(?:;|\n|$)",
            raw_text,
            re.IGNORECASE
        )

        if not match:
            return {
                "status": "error",
                "analysis": f"Invalid SQL generated:\n{raw_text}",
                "chart": None,
                "sql_results": None
            }

        sql_query = match.group(1).strip()

        # -------- EXECUTE SQL --------
        try:
            results = self.database.run(sql_query)
        except Exception as e:
            return {
                "status": "error",
                "analysis": f"SQL Execution Error: {e}",
                "chart": None,
                "sql_results": None
            }

        # -------- VISUALIZATION --------
        df = sql_result_to_dataframe(results)
        chart = (
            create_chart(df, question)
            if df is not None and should_visualize(question)
            else None
        )

        summary = df.to_string(index=False) if df is not None else str(results)

        # -------- ANALYSIS --------
        analysis_response = self.analysis_chain.invoke({
            "question": question,
            "summary": summary
        })

        analysis_text = (
            analysis_response.content
            if hasattr(analysis_response, "content")
            else str(analysis_response)
        )

        return {
            "status": "success",
            "sql_results": results,
            "analysis": analysis_text,
            "chart": chart
        }


# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------
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

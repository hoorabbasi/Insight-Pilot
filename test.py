import pandas as pd
import json
import os
import io
import re
from datetime import datetime

from sqlalchemy import create_engine, inspect
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase

import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY


#chart visualization detection

def should_visualize(question):
    keywords = [
        "chart", "graph", "plot", "visualize", "trend", "compare",
        "distribution", "top", "bottom", "highest", "lowest",
        "month", "year", "sales", "revenue", "customer"
    ]
    return any(word in question.lower() for word in keywords)


#sql result to dataframe

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


# chart generation
def create_chart(df, question):
    q = question.lower()

    if any(w in q for w in ["trend", "month", "year"]):
        return px.line(df, x="label", y="value", title="Trend Analysis")

    if any(w in q for w in ["distribution", "share"]):
        return px.pie(df, names="label", values="value", title="Distribution")

    if any(w in q for w in ["top", "highest", "lowest", "rank"]):
        return px.bar(df, x="value", y="label", orientation="h", title="Ranking")

    return px.bar(df, x="label", y="value", title="Comparison")


# file validation
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


#storing chats

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
    return sorted([f for f in os.listdir("saved_chats") if f.endswith(".json")], reverse=True)


def delete_saved_chat(filename):
    path = os.path.join("saved_chats", filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# AI agent - UPDATED WITHOUT LLMChain

class AnalysisAgent:
    def __init__(self, database, api_key):
        self.database = database

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
            convert_system_message_to_human=True
        )

    def analyze(self, question):
        # Build schema
        schema_text = ""
        for table in self.database.get_usable_table_names():
            schema_text += self.database.get_table_info([table])

        # SQL Generation Prompt
        sql_prompt = f"""You are an AI Data Analyst connected to a SQL database.

STRICT RULES:
1. You MUST always check the database schema.
2. You MUST identify correct tables and columns.
3. You MUST infer column names by meaning.
4. You MUST generate ONE SQLite SELECT query.
5. You MUST NOT use assumptions or general knowledge.
6. You MUST NOT include semicolons.

IMPORTANT COLUMN MATCHING RULE:
- customer / buyer / client → semantic match
- Never fail due to wording mismatch

BEHAVIOR:
- Totals → SUM
- Trends → GROUP BY time
- Rankings → ORDER BY DESC
- Strategy questions → top/bottom performers

CHART REQUIREMENTS:
- Use aliases: SELECT category AS label, metric AS value
- Order logically
- Limit results (Top 5–10)

OUTPUT RULES:
- ONLY SQL
- NO markdown
- NO explanations
- NO semicolons

Database schema:
{schema_text}

Question:
{question}

SQL Query:
"""

        # Generate SQL using direct LLM invoke
        try:
            raw_sql = self.llm.invoke(sql_prompt).content
        except Exception as e:
            return {
                "status": "error",
                "analysis": f"SQL generation failed: {e}",
                "chart": None,
                "sql_results": None
            }

        #SQL SANITIZATION
        match = re.search(r"(SELECT\s+.*)", raw_sql, re.IGNORECASE | re.DOTALL)
        if not match:
            return {
                "status": "error",
                "analysis": f"SQL generation failed:\n{raw_sql}",
                "chart": None,
                "sql_results": None
            }

        sql_query = match.group(1)
        sql_query = sql_query.split(";")[0].strip() 

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

        # Analysis Generation Prompt
        analysis_prompt = f"""You are a business analyst.

Data:
{summary}

Question:
{question}

Respond in 3 sections:
1. What I Found
2. Why It Matters
3. Recommendations
"""

        # Generate analysis using direct LLM invoke
        try:
            analysis = self.llm.invoke(analysis_prompt).content
        except Exception as e:
            analysis = f"Analysis generation failed: {e}"

        return {
            "status": "success",
            "sql_results": results,
            "analysis": analysis,
            "chart": chart
        }


# IMPROVED PDF REPORT GENERATION
def generate_pdf_report(question, sql_results, analysis):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles - simplified without unsupported properties
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=12,
        leading=16
    )
    
    # Build content
    story = []
    
    # Header
    story.append(Paragraph("Business Data Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Question Section
    story.append(Paragraph("<b>Question:</b>", section_heading_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(question, body_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Results Section
    story.append(Paragraph("<b>Query Results:</b>", section_heading_style))
    story.append(Spacer(1, 0.1 * inch))
    
    # Parse and format results into a table
    results_text = str(sql_results)
    try:
        # Try to create a nice table from results
        if '[(' in results_text:
            # Parse tuples
            rows = eval(results_text)
            if rows:
                # Create table data
                table_data = [['Category', 'Value']]
                for row in rows:
                    table_data.append([str(row[0]), str(row[1])])
                
                # Create table
                t = Table(table_data, colWidths=[3*inch, 2*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                ]))
                story.append(t)
            else:
                story.append(Paragraph(results_text, body_style))
        else:
            story.append(Paragraph(results_text, body_style))
    except:
        story.append(Paragraph(results_text, body_style))
    
    story.append(Spacer(1, 0.3 * inch))
    
    # Analysis Section
    story.append(Paragraph("<b>Detailed Analysis:</b>", section_heading_style))
    story.append(Spacer(1, 0.1 * inch))
    
    # Format analysis - simpler approach
    analysis_clean = analysis.replace('**', '<b>').replace('**', '</b>')
    analysis_paragraphs = analysis_clean.split('\n')
    
    for para in analysis_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 0.05 * inch))
    
    story.append(Spacer(1, 0.3 * inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("_" * 80, footer_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Generated by Insight Pilot - AI-Powered Business Analytics", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

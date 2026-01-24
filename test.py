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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


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

        # Format results CLEARLY for analysis - THIS IS THE KEY FIX
        results_str = str(results)
        
        # Create a very clear, unambiguous summary
        formatted_summary = ""
        
        if '[(' in results_str:
            try:
                parsed_results = eval(results_str)
                row_count = len(parsed_results)
                
                formatted_summary = f"""
DATABASE QUERY RESULTS:
======================
Total number of records found: {row_count}

Complete list of ALL {row_count} records:
"""
                for idx, row in enumerate(parsed_results, 1):
                    if len(row) >= 2:
                        formatted_summary += f"\nRecord {idx}: {row[0]} = {row[1]}"
                    else:
                        formatted_summary += f"\nRecord {idx}: {row}"
                
                formatted_summary += f"\n\nIMPORTANT: You must analyze ALL {row_count} records listed above."
                
            except Exception as e:
                formatted_summary = f"Query returned data:\n{results_str}"
        else:
            formatted_summary = f"Query results:\n{results_str}"

        # Analysis Generation Prompt - VERY EXPLICIT
        analysis_prompt = f"""You are a business analyst reviewing database query results.

{formatted_summary}

Original question: {question}

CRITICAL INSTRUCTION: The data above shows MULTIPLE records. You MUST count and analyze ALL records shown above. Do not say "only 1 record" when there are clearly multiple records listed.

Provide your analysis in 3 sections:

1. What I Found
   - State the EXACT number of records found (as shown in "Total number of records found")
   - List ALL items with their values
   - Identify top performers

2. Why It Matters
   - Business implications of these findings
   - Key patterns or trends

3. Recommendations
   - Actionable next steps based on the data

Begin your analysis now:
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


#pdf report 
def generate_pdf_report(question, sql_results, analysis):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20,
        alignment=1,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph("Business Data Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Section heading style
    heading_style = ParagraphStyle(
        'Heading',
        fontSize=12,
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica-Bold',
        spaceAfter=8,
        spaceBefore=15
    )
    
    # Body style
    body_style = ParagraphStyle(
        'Body',
        fontSize=10,
        leading=14,
        spaceAfter=10
    )
    
    # QUESTION
    story.append(Paragraph("QUESTION", heading_style))
    story.append(Paragraph(str(question), body_style))
    story.append(Spacer(1, 0.15*inch))
    
    # RESULTS
    story.append(Paragraph("RESULTS", heading_style))
    results_text = str(sql_results)
    
    # Format results as list
    if '[(' in results_text:
        try:
            results_list = eval(results_text)
            for item in results_list:
                if len(item) >= 2:
                    story.append(Paragraph(f"• {item[0]}: <b>{item[1]}</b>", body_style))
        except:
            story.append(Paragraph(results_text, body_style))
    else:
        story.append(Paragraph(results_text, body_style))
    
    story.append(Spacer(1, 0.15*inch))
    
    # ANALYSIS
    story.append(Paragraph("ANALYSIS", heading_style))
    
    # Parse analysis
    analysis_text = str(analysis).replace('###', '').replace('**', '')
    
    # Split by lines
    lines = analysis_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line == '---':
            continue
        
        # Section headings (1., 2., 3.)
        if len(line) > 0 and line[0].isdigit() and '. ' in line[:4]:
            section_style = ParagraphStyle(
                'Section',
                fontSize=11,
                textColor=colors.HexColor('#34495e'),
                fontName='Helvetica-Bold',
                spaceAfter=6,
                spaceBefore=10
            )
            story.append(Paragraph(line, section_style))
        # Bullet points
        elif line.startswith('* ') or line.startswith('- '):
            bullet_style = ParagraphStyle(
                'Bullet',
                fontSize=10,
                leftIndent=15,
                spaceAfter=5
            )
            bullet_text = line[2:].strip()
            story.append(Paragraph(f"• {bullet_text}", bullet_style))
        # Regular text
        else:
            story.append(Paragraph(line, body_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        fontSize=8,
        textColor=colors.grey,
        alignment=1
    )
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", footer_style))
    story.append(Paragraph("Insight Pilot - AI Business Analytics", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

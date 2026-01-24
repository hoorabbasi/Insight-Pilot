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
        sql_prompt = """You are an AI Data Analyst connected to a SQL database.

STRICT RULES:
1. You MUST always inspect the database schema.
2. You MUST identify and understand the table name(s) and column names before generating any SQL.
3. You MUST infer column names by meaning, not by exact wording.
4. You MUST always generate and execute SQL queries before answering.
5. You MUST base your answers ONLY on the SQL query results.
6. You MUST NOT use general knowledge or assumptions.
7. If the database does not contain enough data to answer, clearly say so.

IMPORTANT COLUMN MATCHING RULE:
- If a column name is not an exact match to the user’s wording, you MUST infer the correct column name by meaning.
- Example: "customer", "buyer", or "client" may refer to columns like customername, client_name, or buyer_name.
- Always choose the closest matching column based on semantic meaning.
- Never fail just because the column name is not an exact text match.

BEHAVIOR:
- Before answering, always decide which table and which columns are relevant.
- If the question asks for numbers (totals, averages, trends), generate SQL and return results.
- If the question asks "why" something happened, analyze historical data, trends, and comparisons from the database.
- If the question asks for suggestions or growth strategies, first analyze the data, then give data-driven recommendations.

CHART REQUIREMENTS:
- Use aliases: SELECT category AS label, metric AS value
- Order logically
- Limit results (Top 5–10)

OUTPUT FORMAT:
- Start with a concise SQL result summary.
- Then explain insights in simple language.
- For "why" questions, explain causes using evidence from the data.
- For "what should we do" questions, provide actionable recommendations backed by data.

Never hallucinate.
Never answer without querying the database.

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


#pdf report 
def generate_pdf_report(question, sql_results, analysis):
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=60,
        leftMargin=60,
        topMargin=60,
        bottomMargin=60
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=1,  # Center
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderPadding=5,
        borderColor=colors.HexColor('#bdc3c7'),
        backColor=colors.HexColor('#ecf0f1')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        fontName='Helvetica'
    )
    
    # Title
    story.append(Paragraph("Business Data Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Question
    story.append(Paragraph("QUESTION", heading_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(str(question), body_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Results
    story.append(Paragraph("QUERY RESULTS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Format results nicely
    results_str = str(sql_results)
    if results_str.startswith('[') and '(' in results_str:
        # Parse and format as a list
        try:
            results_list = eval(results_str)
            for item in results_list:
                if isinstance(item, tuple) and len(item) >= 2:
                    story.append(Paragraph(f"• {item[0]}: {item[1]}", body_style))
        except:
            story.append(Paragraph(results_str, body_style))
    else:
        story.append(Paragraph(results_str, body_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Analysis
    story.append(Paragraph("DETAILED ANALYSIS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Process analysis text
    analysis_text = str(analysis)
    
    # Split into sections
    sections = []
    current_section = []
    
    for line in analysis_text.split('\n'):
        line = line.strip()
        if not line:
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
        else:
            current_section.append(line)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Process each section
    for section in sections:
        if not section.strip():
            continue
            
        # Remove markdown
        section = section.replace('###', '').replace('**', '')
        
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if it's a numbered heading (1., 2., 3.)
            if line[0].isdigit() and '. ' in line[:4]:
                # It's a section heading
                section_heading_style = ParagraphStyle(
                    'SectionHeading',
                    parent=body_style,
                    fontSize=12,
                    textColor=colors.HexColor('#2980b9'),
                    fontName='Helvetica-Bold',
                    spaceAfter=8,
                    spaceBefore=12
                )
                story.append(Paragraph(line, section_heading_style))
            elif line.startswith('* ') or line.startswith('- '):
                # Bullet point
                bullet_text = line[2:].strip()
                bullet_style = ParagraphStyle(
                    'Bullet',
                    parent=body_style,
                    leftIndent=20,
                    spaceAfter=6
                )
                story.append(Paragraph(f"• {bullet_text}", bullet_style))
            else:
                # Regular paragraph
                story.append(Paragraph(line, body_style))
                story.append(Spacer(1, 0.05*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1,  # Center
    )
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", footer_style))
    story.append(Spacer(1, 0.05*inch))
    story.append(Paragraph("Insight Pilot - AI-Powered Business Analytics", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer
